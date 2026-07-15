from __future__ import annotations

import numpy as np

from reference_simulator.engine import execute_batch, initial_state
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Proposal,
    ProposalKind,
    Scenario,
)
from reference_simulator.transition_primitives import validate_proposal
from src.detours.action_suppression import (
    FILTER_DECISION,
    MetricWorseningFilter,
    compare_suppression_pair,
    execute_suppression_arm_summary,
    solve_filtered_graph,
)
from src.detours.barrier_interventions import dynamic_state_digest, make_scenario
from src.detours.barrier_interventions import conditioned_seed, execute_intervention_arm
from src.detours.state_space import FamilySpec, rank_permutation


def _scenario() -> Scenario:
    cells = tuple(
        Cell(
            cell_id=f"c{index}",
            value=index,
            policy=Policy.BUBBLE,
            direction=Direction.ASCENDING,
            fault=FaultMode.NORMAL,
        )
        for index in range(4)
    )
    return Scenario.create(
        cells,
        initial_occupancy=("c0", "c1", "c2", "c3"),
        seed=11,
        max_activations=16,
        architecture=Architecture.CELL_VIEW,
        scheduler="serial_counter_addressed",
        batch_width=1,
        generation_key="s10-test",
        fault_placement="explicit",
        requested_fault_count=0,
    )


def _swap(actor: str, actor_pos: int, target_pos: int) -> Proposal:
    return Proposal(
        kind=ProposalKind.SWAP,
        actor_id=actor,
        actor_pos=actor_pos,
        target_pos=target_pos,
        observed_target_id=f"c{target_pos}",
        reason="test",
        observation_reads=2,
        value_comparisons=1,
    )


def test_filter_suppresses_exact_worsening_and_passes_improvement() -> None:
    scenario = _scenario()
    state = initial_state(scenario)
    worsening = _swap("c0", 0, 1)
    validation = validate_proposal(scenario, state, worsening)
    filter_ = MetricWorseningFilter("inversion_count", 0)
    decision = filter_(scenario, state, worsening, validation, 0)
    assert decision.decision == FILTER_DECISION
    assert decision.eligible_for_commit is False
    audit = filter_.audit_row()
    assert audit["suppressed_count"] == 1
    assert audit["first_suppression_delta"] == 1
    assert audit["filter_decision_exact"] is True

    state.occupancy = ["c1", "c0", "c2", "c3"]
    improving = Proposal(
        kind=ProposalKind.SWAP,
        actor_id="c1",
        actor_pos=0,
        target_pos=1,
        observed_target_id="c0",
        reason="test",
    )
    pass_filter = MetricWorseningFilter("inversion_count", 0)
    passed = pass_filter(
        scenario,
        state,
        improving,
        validate_proposal(scenario, state, improving),
        0,
    )
    assert passed.eligible_for_commit is True
    assert pass_filter.audit_row()["eligible_improving_count"] == 1


def test_filter_passes_native_rejections_and_threshold_one() -> None:
    scenario = _scenario()
    state = initial_state(scenario)
    proposal = _swap("c0", 0, 1)
    tolerant = MetricWorseningFilter("inversion_count", 1)
    decision = tolerant(
        scenario, state, proposal, validate_proposal(scenario, state, proposal), 0
    )
    assert decision.eligible_for_commit is True
    audit = tolerant.audit_row()
    assert audit["eligible_allowed_worsening_count"] == 1
    assert audit["suppressed_count"] == 0

    invalid = Proposal(
        kind=ProposalKind.SWAP,
        actor_id="c0",
        actor_pos=0,
        target_pos=99,
        reason="invalid",
    )
    rejected = tolerant(
        scenario, state, invalid, validate_proposal(scenario, state, invalid), 1
    )
    assert rejected.decision == "rejected_invalid_target"
    assert tolerant.audit_row()["native_ineligible_count"] == 1


def test_engine_filter_changes_only_decision_commit_and_outcome_cost() -> None:
    family = FamilySpec(
        n=4,
        architecture=Architecture.CELL_VIEW,
        direction=Direction.ASCENDING,
        policies=(Policy.BUBBLE,) * 4,
        faults=(FaultMode.NORMAL,) * 4,
    )
    scenario = make_scenario(
        family, seed=3, max_activations=8, generation_key="s10-engine-test"
    )
    control = initial_state(scenario)
    filtered = initial_state(scenario)

    def factory(scenario, state, actor_id, *, side=None):
        return _swap("c0", 0, 1)

    control_events, _ = execute_batch(
        scenario, control, retain_events=True, proposal_factory=factory
    )
    filter_ = MetricWorseningFilter("inversion_count", 0)
    filtered_events, _ = execute_batch(
        scenario,
        filtered,
        retain_events=True,
        proposal_factory=factory,
        proposal_validation_filter=filter_,
    )
    assert control_events[0]["proposal"] == filtered_events[0]["proposal"]
    assert control_events[0]["preStateHash"] == filtered_events[0]["preStateHash"]
    assert control_events[0]["decision"] == "accepted"
    assert filtered_events[0]["decision"] == FILTER_DECISION
    assert control.occupancy == ["c1", "c0", "c2", "c3"]
    assert filtered.occupancy == ["c0", "c1", "c2", "c3"]
    assert filtered.ledger["rejections"] == 1
    assert filtered.ledger["acceptedSwaps"] == 0


def test_filtered_graph_distinguishes_created_impossibility() -> None:
    # 0 -> 1 is a required local worsening, then 1 -> 2 reaches the goal.
    source = np.array([0, 1, 0], dtype=np.int64)
    target = np.array([1, 2, 0], dtype=np.int64)
    decisions = np.array([1, 1, 0], dtype=np.int8)
    deltas = np.array([1, -2, 0], dtype=np.int8)
    levels = np.array([1, 2, 0], dtype=np.int8)
    terminal = np.array([0, 0, 1], dtype=np.uint8)
    strict, suppressed = solve_filtered_graph(
        3, source, target, decisions, deltas, levels, terminal, 0
    )
    assert suppressed.tolist() == [True, False, False]
    assert strict.classifications[0] == 3  # unreachable_active
    tolerant, _ = solve_filtered_graph(
        3, source, target, decisions, deltas, levels, terminal, 1
    )
    assert tolerant.classifications[0] == 2  # necessary_detour


def test_failed_shorter_run_has_no_efficiency_delta() -> None:
    base_trace = [
        {
            "actor_id": "c0",
            "raw_side_value": 1,
            "side_consumed": True,
            "proposal_kind": "Swap",
            "proposal_reason": "x",
            "proposal_actor_pos": 0,
            "proposal_target_pos": 1,
            "proposal_new_cursor": None,
            "proposal_observation_reads": 2,
            "proposal_value_comparisons": 1,
            "decision": "accepted",
            "before_dynamic_hash": "a",
            "after_dynamic_hash": "b",
            "before_occupancy": ("c0", "c1"),
            "after_occupancy": ("c1", "c0"),
            "before_levels": {name: 0 for name in ("adjacent_descents", "inversion_count", "spearman_footrule", "maximum_rank_error")},
            "after_levels": {name: 1 for name in ("adjacent_descents", "inversion_count", "spearman_footrule", "maximum_rank_error")},
            "ledger_delta": {},
        }
    ]
    filtered_trace = [dict(base_trace[0], decision=FILTER_DECISION, after_dynamic_hash="a", after_occupancy=("c0", "c1"))]
    common = {
        "pre_dynamic_state_sha256": "a",
        "scenario_id": "s",
        "cost_activations": 1,
        "cost_acceptedSwaps": 1,
        "full_ledger_unit_cost": 5,
        **{f"peak_{name}": 1 for name in ("adjacent_descents", "inversion_count", "spearman_footrule", "maximum_rank_error")},
        **{f"excursion_{name}": 1 for name in ("adjacent_descents", "inversion_count", "spearman_footrule", "maximum_rank_error")},
        **{f"final_{name}": 0 for name in ("adjacent_descents", "inversion_count", "spearman_footrule", "maximum_rank_error")},
    }
    control = dict(common, completed=True, stop_reason="complete")
    filtered = dict(common, completed=False, stop_reason="quiescent", cost_activations=0, cost_acceptedSwaps=0, full_ledger_unit_cost=0)
    audit = {"first_suppression_event": 0, "suppressed_count": 1}
    result = compare_suppression_pair(control, base_trace, filtered, filtered_trace, audit)
    assert result["delta_spent_activations"] == -1
    assert result["efficiency_comparable"] is False
    assert result["efficiency_delta_activations"] is None


def test_summary_runner_matches_full_engine_after_divergence() -> None:
    family = FamilySpec(
        n=4,
        architecture=Architecture.CELL_VIEW,
        direction=Direction.ASCENDING,
        policies=(Policy.BUBBLE,) * 4,
        faults=(FaultMode.NORMAL, FaultMode.NORMAL, FaultMode.NORMAL, FaultMode.STUCK),
    )
    state_ordinal = rank_permutation((2, 0, 1, 3))
    key = "s10-summary-parity"
    seed, attempts = conditioned_seed(key, 3, 4, 0)
    full_filter = MetricWorseningFilter("inversion_count", 0)
    full, _ = execute_intervention_arm(
        family,
        state_ordinal,
        source_family_ordinal=1,
        arm_family_ordinal=1,
        intervention_type="baseline",
        arm_variant="baseline",
        focal_index=3,
        target_index=None,
        replicate_index=0,
        coupling_key=key,
        coupling_seed=seed,
        seed_search_attempts=attempts,
        max_activations=64,
        proposal_validation_filter=full_filter,
        trace_proposal_details=True,
    )
    summary, prefix, summary_filter, actors, sides, consumption = (
        execute_suppression_arm_summary(
            family,
            state_ordinal,
            source_family_ordinal=1,
            arm_family_ordinal=1,
            intervention_type="baseline",
            arm_variant="baseline",
            focal_index=3,
            target_index=None,
            replicate_index=0,
            coupling_key=key,
            coupling_seed=seed,
            seed_search_attempts=attempts,
            maximum_activations=64,
            metric="inversion_count",
            threshold=0,
        )
    )
    fields = [
        "scenario_id",
        "stop_reason",
        "completed",
        "event_count",
        "final_state_ordinal",
        "final_dynamic_state_sha256",
        "full_ledger_unit_cost",
    ]
    fields.extend(key for key in full if key.startswith("cost_"))
    fields.extend(
        key
        for key in full
        if key.startswith(("start_", "peak_", "excursion_", "final_", "worsening_events_"))
    )
    assert all(summary[field] == full[field] for field in fields)
    assert summary_filter.audit_row() == full_filter.audit_row()
    assert len(actors) == len(sides) == len(consumption) == summary["event_count"]
    assert len(prefix) == summary["native_event_prefix_count"]
    assert summary["summary_complete"] is True
