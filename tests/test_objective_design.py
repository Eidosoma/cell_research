from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.objective_design import (
    bin_index,
    build_s04_evidence,
    load_yaml,
    require_s05_eligible,
)


CONFIG = Path("configs/search")
S02 = Path("/artifacts/research_steps/S02")
S03 = Path("/artifacts/research_steps/S03")


@pytest.fixture(scope="module")
def evidence():
    return build_s04_evidence(
        objective_path=CONFIG / "s04_objective_registry.yaml",
        descriptor_path=CONFIG / "s04_descriptor_registry.yaml",
        weighting_path=CONFIG / "s04_task_weighting.yaml",
        uncertainty_path=CONFIG / "s04_uncertainty_registry.yaml",
        anti_gaming_path=CONFIG / "s04_anti_gaming_registry.yaml",
        eligibility_path=CONFIG / "s04_s05_eligibility_gate.yaml",
        baseline_smoke_path=S02 / "baseline_smoke_results.json",
        split_manifest_path="configs/environment_suite/split_manifest.json",
        seed_evaluations_path=S03 / "baseline_evaluations.jsonl",
        complexity_path=S03 / "complexity_accounting.jsonl",
    )


def test_bin_contract_is_lower_inclusive_and_final_upper_inclusive():
    edges = [0.0, 0.5, 1.0]
    assert bin_index(0.0, edges) == 0
    assert bin_index(0.499, edges) == 0
    assert bin_index(0.5, edges) == 1
    assert bin_index(1.0, edges) == 1
    with pytest.raises(ValueError):
        bin_index(-0.1, edges)
    with pytest.raises(ValueError):
        bin_index(float("nan"), edges)


def test_objective_registry_forbids_universal_scores_and_preserves_special_costs():
    registry = load_yaml(CONFIG / "s04_objective_registry.yaml")
    rules = registry["globalRules"]
    assert not rules["universalNormalizedScorePermitted"]
    assert not rules["crossTaskScalarizationPermitted"]
    assert not rules["crossTaskDominancePermitted"]
    common = registry["commonObjectiveFamilies"]
    assert "licensedPrefixValueReads" in {
        row["field"] for row in common["licensedInsertionPrefix"]["fields"]
    }
    assert "licensedLongRangeRequestedDistance" in {
        row["field"] for row in common["licensedSelectionCursorAndRange"]["fields"]
    }
    assert common["licensedInsertionPrefix"]["aggregation"].startswith("separate")
    assert common["licensedSelectionCursorAndRange"]["aggregation"].startswith(
        "separate"
    )


def test_descriptor_registry_excludes_s03_and_unbound_fixture_shortcuts():
    registry = load_yaml(CONFIG / "s04_descriptor_registry.yaml")
    policy = registry["archivePolicy"]
    assert not policy["s03StructuralCoverageUsedAsDescriptor"]
    assert not policy["s03FiniteCorpusEquivalenceUsedAsDescriptor"]
    assert not policy["e05UnboundRowsUsedAsDescriptor"]
    assert not policy["e06SyntheticTypedFixturesUsedAsDescriptor"]
    assert not policy["validationOutcomesUsed"]
    assert not policy["confirmationOutcomesUsed"]


def test_s05_gate_requires_e05_e06_and_is_currently_blocked():
    gate = load_yaml(CONFIG / "s04_s05_eligibility_gate.yaml")
    assert gate["e05AdapterRequiredBeforeS05"]
    assert gate["e06AdapterRequiredBeforeS05"]
    assert not gate["fieldNameSimilarityCanWaiveAdapter"]
    assert not gate["s05SearchEligible"]
    statuses = {row["id"]: row["currentStatus"] for row in gate["requirements"]}
    assert statuses["G03_E05_arbitrary_DSL_binding"] == "fail"
    assert statuses["G04_E06_arbitrary_DSL_binding"] == "fail"
    with pytest.raises(RuntimeError, match="S05 search eligibility gate is blocked"):
        require_s05_eligible(gate)


def test_training_only_evidence_and_adversarial_coverage(evidence):
    assert evidence["success"]
    assert len(evidence["normalizationBaselines"]["taskBaselines"]) == 8
    assert {
        row["split"] for row in evidence["normalizationBaselines"]["taskBaselines"]
    } == {"train"}
    assert evidence["normalizationBaselines"]["validationOutcomeEvaluations"] == 0
    assert evidence["normalizationBaselines"]["confirmationOutcomeEvaluations"] == 0
    assert evidence["adversarialAudit"]["passed"]
    assert not evidence["adversarialAudit"]["unexercisedRegisteredChannels"]


def test_descriptor_bounds_occupancy_and_stability(evidence):
    validation = evidence["descriptorValidation"]
    assert validation["passed"]
    assert validation["numericConformanceFixtureCellCount"] == 100
    assert not validation["numericConformanceFixturesArePolicyOrEfficacyEvidence"]
    regeneration = next(
        row
        for row in validation["nativeTrainingBaselineDescriptorRows"]
        if row["taskId"] == "e07_s02_regeneration_1d"
    )
    assert not regeneration["available"]
    spatial = [
        row
        for row in validation["nativeTrainingBaselineDescriptorRows"]
        if row["taskId"].startswith("e07_s02_spatial")
    ]
    assert all("not_S03_fixture" in row["provenance"] for row in spatial)


def test_correlations_are_non_efficacy_and_task_weights_do_not_rank(evidence):
    correlation = evidence["correlationSummary"]
    assert correlation["policyCount"] == 30
    assert "not efficacy" in correlation["scope"]
    assert not correlation["primaryNativeDescriptorCorrelationEstimable"]
    assert correlation["noCrossTaskPerformanceCorrelationComputed"]
    weights = evidence["weightingSensitivity"]
    assert weights["passed"]
    assert not weights["policyRankingComputed"]
    assert not weights["weightedPerformanceScoreComputed"]
    assert set(weights["s05EligibilityByProfile"].values()) == {False}


def test_evidence_rebuild_is_deterministic(evidence):
    second = build_s04_evidence(
        objective_path=CONFIG / "s04_objective_registry.yaml",
        descriptor_path=CONFIG / "s04_descriptor_registry.yaml",
        weighting_path=CONFIG / "s04_task_weighting.yaml",
        uncertainty_path=CONFIG / "s04_uncertainty_registry.yaml",
        anti_gaming_path=CONFIG / "s04_anti_gaming_registry.yaml",
        eligibility_path=CONFIG / "s04_s05_eligibility_gate.yaml",
        baseline_smoke_path=S02 / "baseline_smoke_results.json",
        split_manifest_path="configs/environment_suite/split_manifest.json",
        seed_evaluations_path=S03 / "baseline_evaluations.jsonl",
        complexity_path=S03 / "complexity_accounting.jsonl",
    )
    assert json.dumps(evidence, sort_keys=True) == json.dumps(second, sort_keys=True)
