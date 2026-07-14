"""Focused validation for the E02 S01 frozen causal design."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from analysis.causal_model import (
    compatibility_rows,
    generate,
    registry,
    validate_design,
)


def test_registry_passes_all_design_checks() -> None:
    result = validate_design(registry())
    assert result["success"]
    assert all(check["passed"] for check in result["checks"].values())
    assert result["counts"]["primaryEstimands"] == 12
    assert result["counts"]["descriptiveEstimands"] == 1


def test_primary_adjustment_sets_are_baseline_only() -> None:
    value = registry()
    primary = [row for row in value["estimands"] if row["tier"].startswith("primary")]
    assert primary
    assert {item for row in primary for item in row["adjustmentSet"]} == {
        "X_SCENARIO:pairing_block"
    }
    prohibited = " ".join(value["mediatorPolicy"]["prohibited"])
    assert "Do not adjust" in prohibited
    assert "completion" in prohibited


def test_legacy_bundle_is_not_called_pure_architecture() -> None:
    value = registry()
    descriptive = [row for row in value["estimands"] if row["tier"] == "descriptive_anchor"]
    assert len(descriptive) == 1
    assert descriptive[0]["effectType"] == "descriptive_not_pure_architecture"
    assert descriptive[0]["identification"].startswith("not identified")


def test_all_unimplemented_levels_have_gates() -> None:
    rows = compatibility_rows(registry())
    pending = [row for row in rows if "not_implemented" in row["implementation_status"]]
    assert pending
    assert all(row["implementation_gate"] != "none" for row in pending)


def test_generator_writes_parseable_required_artifacts(tmp_path: Path) -> None:
    result = generate(tmp_path)
    assert result["success"]
    required = {
        "causal_dag.svg",
        "estimand_registry.yaml",
        "analysis_population.md",
        "intervention_compatibility.csv",
        "validation_summary.json",
        "provenance_manifest.json",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    parsed = yaml.safe_load((tmp_path / "estimand_registry.yaml").read_text())
    assert parsed["researchStepId"] == "S01"
    validation = json.loads((tmp_path / "validation_summary.json").read_text())
    assert validation["success"]
    svg = (tmp_path / "causal_dag.svg").read_text()
    assert svg.startswith("<svg")
    assert "post-treatment mediators" in svg
