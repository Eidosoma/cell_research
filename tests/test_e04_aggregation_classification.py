from __future__ import annotations

import json
from pathlib import Path

from analysis.aggregation_classification import (
    CLASSIFICATIONS,
    build_benchmarks,
    build_claims,
    build_decision_audit,
    build_terminology_audit,
    extract_facts,
    resolve_benchmark_rows,
    terminology_scan,
    validate_upstream_manifests,
)


ARTIFACT_ROOT = Path("/artifacts")
REPO = Path(__file__).resolve().parents[1]


def test_contract_has_frozen_eight_gate_rule() -> None:
    contract = json.loads(
        (REPO / "analysis" / "s14_aggregation_classification_contract.json").read_text()
    )
    assert contract["researchStepId"] == "S14"
    assert contract["frozenBeforeSynthesis"] is True
    assert len(contract["decisionGates"]) == 8
    assert set(contract["classificationVocabulary"]) == CLASSIFICATIONS


def test_facts_and_claims_are_complete_and_deterministic() -> None:
    facts = extract_facts(ARTIFACT_ROOT)
    first = build_claims(facts)
    second = build_claims(extract_facts(ARTIFACT_ROOT))
    assert first == second
    assert len(first) == 22
    assert len({row["claimId"] for row in first}) == 22
    assert set(row["classification"] for row in first) == CLASSIFICATIONS
    assert facts["paperProfilesMatchingPeakAndTiming"] == 3
    assert facts["balancedDynamicPeakPassed"] == 6
    assert (
        facts["phaseModalFixed"]
        + facts["phaseModalDynamic"]
        + facts["phaseModalDominance"]
        + facts["phaseModalUnresolved"]
        == 4500
    )


def test_decision_rules_and_terminology_have_full_coverage() -> None:
    contract = json.loads(
        (REPO / "analysis" / "s14_aggregation_classification_contract.json").read_text()
    )
    claims = build_claims(extract_facts(ARTIFACT_ROOT))
    audit = build_decision_audit(claims, contract)
    terms = build_terminology_audit(contract)
    assert len(audit) == 8
    assert all(row["covered"] for row in audit)
    assert any(row["term"] == "fixed quiescence" for row in terms)
    assert any(
        row["term"] == "operational attractor" and not row["positiveClaimAllowed"]
        for row in terms
    )


def test_benchmark_selectors_resolve_and_cover_phase_states() -> None:
    rows = resolve_benchmark_rows(build_benchmarks(ARTIFACT_ROOT))
    assert len(rows) == 21
    assert all(row["resolved"] for row in rows)
    phase = [row for row in rows if row["family"] == "phase_exemplar"]
    assert len(phase) == 4
    assert {row["expectedDecision"] for row in phase} == {
        "fixed_quiescence",
        "dynamic_equilibrium",
        "active_dominance",
        "unresolved_transient",
    }


def test_upstream_manifests_are_intact() -> None:
    result = validate_upstream_manifests(ARTIFACT_ROOT)
    assert result["passed"]
    assert len(result["steps"]) == 14
    assert result["manifestRecords"] > 400


def test_terminology_scanner_rejects_positive_prohibited_use(tmp_path: Path) -> None:
    contract = json.loads(
        (REPO / "analysis" / "s14_aggregation_classification_contract.json").read_text()
    )
    good = tmp_path / "good.md"
    good.write_text("No operational attractor is claimed.\n", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("This is an operational attractor.\n", encoding="utf-8")
    assert terminology_scan([good], contract)["passed"]
    assert not terminology_scan([bad], contract)["passed"]
