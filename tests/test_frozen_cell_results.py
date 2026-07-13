from __future__ import annotations

from pathlib import Path

from analysis.frozen_cell_results import (
    CONFIRM_R_COUNT,
    PAPER_C_COUNT,
    PAPER_R_COUNT,
    hand_counted_fault_toys,
    load_tasks,
    monotonicity_error,
    preregistration_sha256,
    run_reference_summary,
)
from reference_simulator.api import create_scenario, run_scenario
from reference_simulator.model import LEDGER_FIELDS


def test_s11_preregistration_and_population_gates() -> None:
    digest = preregistration_sha256()
    assert len(digest) == 64
    assert len(load_tasks("paper_scale", "reference")) == PAPER_R_COUNT
    assert len(load_tasks("paper_scale", "historical")) == PAPER_C_COUNT
    confirm = load_tasks(
        "confirmatory_holdout", "reference", preregistration_hash=digest
    )
    assert len(confirm) == CONFIRM_R_COUNT
    assert all(task["scenario_row"]["protected"] for task in confirm)
    assert {
        task["scenario_row"]["placementProfile"] for task in confirm
    } == {"reference_without_replacement"}


def test_s11_fast_summary_matches_digest_across_fault_semantics() -> None:
    scenarios = []
    for architecture in ("cell_view", "traditional"):
        for policy in ("Bubble", "Insertion", "Selection"):
            for fault in ("passive", "stuck"):
                scenarios.append(
                    create_scenario(
                        [5, 1, 4, 2, 3],
                        policy=policy,
                        architecture=architecture,
                        faults={1: fault, 3: fault},
                        generation_key=f"S11/test/{architecture}/{policy}/{fault}",
                        permute=False,
                        seed=111,
                        max_activations=5000,
                    )
                )
    for scenario in scenarios:
        ordinary = run_scenario(scenario, trace_mode="digest")
        fast = run_reference_summary(scenario)
        assert fast["final_state_hash"] == ordinary.final_state_hash
        assert fast["stop_reason"] == ordinary.summary["stopReason"]
        assert fast["activation_count"] == ordinary.summary["activationCount"]
        assert fast["monotonicity_error"] == monotonicity_error(
            ordinary.summary["finalValues"]
        )
        assert all(
            fast[f"ledger_{field}"] == ordinary.summary["ledger"][field]
            for field in LEDGER_FIELDS
        )


def test_s11_fault_semantics_toys_pass() -> None:
    result = hand_counted_fault_toys()
    assert result["success"]
    assert result["passed"] == result["total"] == 6

