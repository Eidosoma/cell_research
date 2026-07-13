import hashlib
import json
from pathlib import Path

import pytest

from analysis.no_fault_sorting import (
    PREREGISTRATION,
    load_tasks,
    normalized_curve,
    run_reference_summary,
    sortedness_percent,
)
from reference_simulator.engine import execute_batch, initial_state, run
from reference_simulator.model import Architecture, Cell, Direction, Policy, Scenario


def scenario(policy=Policy.BUBBLE, architecture=Architecture.CELL_VIEW):
    cells = tuple(
        Cell(f"c{index}", value, policy, Direction.ASCENDING)
        for index, value in enumerate([3, 1, 2])
    )
    return Scenario.create(
        cells,
        architecture=architecture,
        traditional_policy=policy if architecture == Architecture.TRADITIONAL else None,
        seed=17,
        max_activations=10_000,
        generation_key=f"S09-test-{policy.value}-{architecture.value}",
    )


def test_paper_sortedness_and_normalized_curve():
    assert sortedness_percent([3, 1, 2]) == pytest.approx(200 / 3)
    assert sortedness_percent([1, 2, 3]) == 100
    curve = normalized_curve([[0, 50], [2, 100]])
    assert len(curve) == 201
    assert curve[100] == 75


@pytest.mark.parametrize("architecture", list(Architecture))
@pytest.mark.parametrize("policy", list(Policy))
def test_summary_only_path_matches_digest_path(architecture, policy):
    item = scenario(policy, architecture)
    summary = run_reference_summary(item, retain_raw_curve=True)
    ordinary = run(item, trace_mode="digest")
    assert summary["completed"]
    assert summary["final_state_hash"] == ordinary.final_state_hash
    assert summary["activation_count"] == ordinary.summary["activationCount"]
    assert summary["successful_swap_count"] == ordinary.summary["ledger"]["acceptedSwaps"]
    assert summary["final_sortedness_percent"] == 100


def test_execute_batch_rejects_retention_without_emission():
    item = scenario()
    with pytest.raises(ValueError, match="retained events require"):
        execute_batch(item, initial_state(item), retain_events=True, emit_event_records=False)


def test_protected_split_gate_and_exact_unlock():
    with pytest.raises(PermissionError):
        load_tasks("exploratory")
    with pytest.raises(PermissionError):
        load_tasks("policy_search_holdout")
    with pytest.raises(PermissionError):
        load_tasks("confirmatory_holdout", preregistration_sha256="wrong")
    digest = hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    tasks = load_tasks("confirmatory_holdout", preregistration_sha256=digest)
    assert len(tasks) == 6000
    assert {task["scenario_row"]["split"] for task in tasks} == {"confirmatory_holdout"}
    assert sum(task["retain_raw_curve"] for task in tasks) == 6
