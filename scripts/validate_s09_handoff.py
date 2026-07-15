#!/usr/bin/env python3
"""Validate the final S09 report and workflow handoff without rerunning compute."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/artifacts/research_steps/S09")
PLAN = Path("/workspace/RESEARCH_PLAN.md")


def main() -> None:
    report = (ROOT / "research_step_full_results.md").read_text()
    plan = PLAN.read_text()
    phrases = (
        "Research Step S09 Full Results — Perform Barrier Interventions",
        "Completion status",
        "Artifacts written",
        "Validation result",
        "Outcome classification",
        "Caveats or blockers",
        "Recommended next action",
        "### Lay summary",
        "## Frozen inputs",
        "## Methods",
        "## Results",
        "## Validation",
        "## Commands",
        "## Artifacts",
        "## Provenance",
    )
    required = (
        "research_step_full_results.md",
        "barrier_interventions.parquet",
        "paired_effects.parquet",
        "paired_metric_effects.parquet",
        "effect_summary.parquet",
        "paired_traces.parquet",
        "event_budget_sensitivity.parquet",
        "validation_results.json",
        "provenance_manifest.json",
        "artifact_manifest.json",
        "result_summary.json",
    )
    validation = json.loads((ROOT / "validation_results.json").read_text())
    gates = {
        "required_outputs": all(
            (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0
            for name in required
        ),
        "markdown_handoff_contract": all(phrase in report for phrase in phrases),
        "scientific_validation_passed": validation["passed"] is True,
        "plan_marks_s09_complete": "| S09 | Perform barrier interventions | Complete (2026-07-15)" in plan,
        "plan_advances_only_to_s10": "Step ID: S10 (next; not started)" in plan,
        "report_stops_before_s10": "No S10 artifact directory was created" in report,
        "filesystem_has_no_s10": not Path("/artifacts/research_steps/S10").exists(),
    }
    value = {
        "schemaVersion": "e03.s09.handoff_validation.v1",
        "researchStepId": "S09",
        "passed": all(gates.values()),
        "gateCount": len(gates),
        "passedGateCount": sum(gates.values()),
        "gates": gates,
    }
    (ROOT / "handoff_validation.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps(value, indent=2))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
