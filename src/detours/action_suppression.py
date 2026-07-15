"""Transparent metric-specific proposal suppression for E03 S10.

The intervention runs after authoritative proposal construction and mechanical
validation.  It rejects exactly those otherwise eligible proposals whose
committed successor would increase the selected exact distance numerator by
more than a declared threshold.  It never changes the scenario, scheduler,
proposal, native validation, commit implementation, ledger, or terminal rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from reference_simulator.engine import EMPTY_DIGEST, execute_batch, evaluate_terminal
from reference_simulator.model import (
    Architecture,
    Proposal,
    RunState,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.transition_primitives import (
    ValidationDecision,
    commit_proposal,
    ledger_identity,
)
from src.detours.barrier_interventions import (
    METRICS,
    _CoupledSchedule,
    coupled_token,
    dynamic_state_digest,
    make_scenario,
    metric_levels,
    structural_state_from_ordinal,
)
from src.detours.necessary_detour import CLASS_LABELS, solve_primary_all_starts
from src.detours.state_space import FamilySpec, StructuralState


SCHEMA_VERSION = "e03.s10.action_suppression.v1"
FILTER_DECISION = "rejected_metric_worsening"
METRIC_DELTA_COLUMNS = {
    "adjacent_descents": "delta_adjacent_descents",
    "inversion_count": "delta_inversion_count",
    "spearman_footrule": "delta_spearman_footrule",
    "maximum_rank_error": "delta_maximum_rank_error",
}


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(slots=True)
class MetricWorseningFilter:
    """Audit-complete one-step exact-distance proposal filter."""

    metric: str
    threshold: int = 0
    calls: int = 0
    native_eligible: int = 0
    native_ineligible: int = 0
    eligible_improving: int = 0
    eligible_neutral: int = 0
    eligible_allowed_worsening: int = 0
    eligible_excess_worsening: int = 0
    suppressed: int = 0
    false_positive: int = 0
    false_negative: int = 0
    first_suppression_event: int | None = None
    first_suppression_delta: int | None = None
    first_suppression_before: int | None = None
    first_suppression_candidate_after: int | None = None
    first_suppression_proposal_sha256: str | None = None
    maximum_suppressed_delta: int | None = None
    total_suppressed_delta: int = 0

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"unsupported S10 metric: {self.metric}")
        if self.threshold < 0:
            raise ValueError("S10 suppression threshold must be nonnegative")

    def __call__(
        self,
        scenario: Scenario,
        state: RunState,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision:
        self.calls += 1
        if not validation.eligible_for_commit:
            self.native_ineligible += 1
            return validation

        self.native_eligible += 1
        before = metric_levels(scenario, state)[self.metric]
        candidate = state.clone()
        changed = commit_proposal(candidate, state, proposal, "accepted")
        if not changed:
            raise AssertionError("eligible proposal did not commit in S10 audit clone")
        after = metric_levels(scenario, candidate)[self.metric]
        delta = int(after - before)
        if delta < 0:
            self.eligible_improving += 1
        elif delta == 0:
            self.eligible_neutral += 1
        elif delta <= self.threshold:
            self.eligible_allowed_worsening += 1
        else:
            self.eligible_excess_worsening += 1

        should_suppress = delta > self.threshold
        if not should_suppress:
            return validation

        self.suppressed += 1
        self.total_suppressed_delta += delta
        self.maximum_suppressed_delta = (
            delta
            if self.maximum_suppressed_delta is None
            else max(self.maximum_suppressed_delta, delta)
        )
        if self.first_suppression_event is None:
            self.first_suppression_event = int(event_index)
            self.first_suppression_delta = delta
            self.first_suppression_before = int(before)
            self.first_suppression_candidate_after = int(after)
            self.first_suppression_proposal_sha256 = canonical_digest(
                proposal.to_dict()
            )
        return ValidationDecision(FILTER_DECISION, False)

    def audit_row(self) -> dict[str, Any]:
        partition = (
            self.eligible_improving
            + self.eligible_neutral
            + self.eligible_allowed_worsening
            + self.eligible_excess_worsening
        )
        self.false_negative = self.eligible_excess_worsening - self.suppressed
        return {
            "filter_metric": self.metric,
            "filter_threshold": self.threshold,
            "audited_event_count": self.calls,
            "native_eligible_count": self.native_eligible,
            "native_ineligible_count": self.native_ineligible,
            "eligible_improving_count": self.eligible_improving,
            "eligible_neutral_count": self.eligible_neutral,
            "eligible_allowed_worsening_count": self.eligible_allowed_worsening,
            "eligible_excess_worsening_count": self.eligible_excess_worsening,
            "suppressed_count": self.suppressed,
            "false_positive_count": self.false_positive,
            "false_negative_count": self.false_negative,
            "event_partition_valid": self.calls
            == self.native_eligible + self.native_ineligible,
            "eligible_partition_valid": self.native_eligible == partition,
            "filter_decision_exact": self.suppressed
            == self.eligible_excess_worsening,
            "first_suppression_event": self.first_suppression_event,
            "first_suppression_delta": self.first_suppression_delta,
            "first_suppression_before": self.first_suppression_before,
            "first_suppression_candidate_after": self.first_suppression_candidate_after,
            "first_suppression_proposal_sha256": self.first_suppression_proposal_sha256,
            "maximum_suppressed_delta": self.maximum_suppressed_delta,
            "total_suppressed_delta": self.total_suppressed_delta,
        }


def _first_difference(
    control_trace: Sequence[Mapping[str, Any]],
    filtered_trace: Sequence[Mapping[str, Any]],
    field: str,
) -> int | None:
    common = min(len(control_trace), len(filtered_trace))
    for index in range(common):
        if control_trace[index][field] != filtered_trace[index][field]:
            return index
    if len(control_trace) != len(filtered_trace):
        return common
    return None


def compare_suppression_pair(
    control_row: Mapping[str, Any],
    control_trace: Sequence[Mapping[str, Any]],
    filtered_row: Mapping[str, Any],
    filtered_trace: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Return pairing and pre-divergence audits plus outcome contrasts."""

    common = min(len(control_trace), len(filtered_trace))
    actor_tokens_equal = all(
        control_trace[index]["actor_id"] == filtered_trace[index]["actor_id"]
        for index in range(common)
    )
    side_tokens_equal = all(
        control_trace[index]["raw_side_value"]
        == filtered_trace[index]["raw_side_value"]
        for index in range(common)
    )
    side_consumption_equal = all(
        control_trace[index]["side_consumed"]
        == filtered_trace[index]["side_consumed"]
        for index in range(common)
    )
    first_decision = _first_difference(control_trace, filtered_trace, "decision")
    first_dynamic = _first_difference(
        control_trace, filtered_trace, "after_dynamic_hash"
    )
    first_suppression = audit["first_suppression_event"]
    pre_divergence_fields = (
        "actor_id",
        "raw_side_value",
        "side_consumed",
        "proposal_kind",
        "proposal_reason",
        "proposal_actor_pos",
        "proposal_target_pos",
        "proposal_new_cursor",
        "proposal_observation_reads",
        "proposal_value_comparisons",
        "decision",
        "before_dynamic_hash",
        "after_dynamic_hash",
        "before_occupancy",
        "after_occupancy",
        "before_levels",
        "after_levels",
        "ledger_delta",
    )
    prefix_end = common if first_decision is None else first_decision
    pre_divergence_identity = all(
        control_trace[index][field] == filtered_trace[index][field]
        for index in range(prefix_end)
        for field in pre_divergence_fields
    )
    divergence_proposal_equal = None
    divergence_prestate_equal = None
    divergence_decision_is_filter_only = None
    if first_decision is not None and first_decision < common:
        proposal_fields = (
            "actor_id",
            "raw_side_value",
            "side_consumed",
            "proposal_kind",
            "proposal_reason",
            "proposal_actor_pos",
            "proposal_target_pos",
            "proposal_new_cursor",
            "proposal_observation_reads",
            "proposal_value_comparisons",
        )
        divergence_proposal_equal = all(
            control_trace[first_decision][field]
            == filtered_trace[first_decision][field]
            for field in proposal_fields
        )
        divergence_prestate_equal = (
            control_trace[first_decision]["before_dynamic_hash"]
            == filtered_trace[first_decision]["before_dynamic_hash"]
            and control_trace[first_decision]["before_levels"]
            == filtered_trace[first_decision]["before_levels"]
        )
        divergence_decision_is_filter_only = (
            control_trace[first_decision]["decision"] == "accepted"
            and filtered_trace[first_decision]["decision"] == FILTER_DECISION
        )

    no_suppression_full_trace_equal = None
    if int(audit["suppressed_count"]) == 0:
        no_suppression_full_trace_equal = (
            len(control_trace) == len(filtered_trace)
            and all(
                control_trace[index] == filtered_trace[index]
                for index in range(len(control_trace))
            )
        )

    result: dict[str, Any] = {
        "common_prefix_event_count": common,
        "pre_dynamic_state_equal": control_row["pre_dynamic_state_sha256"]
        == filtered_row["pre_dynamic_state_sha256"],
        "scenario_id_equal": control_row["scenario_id"]
        == filtered_row["scenario_id"],
        "actor_token_common_prefix_equal": actor_tokens_equal,
        "side_token_common_prefix_equal": side_tokens_equal,
        "side_consumption_common_prefix_equal": side_consumption_equal,
        "first_decision_divergence_event": first_decision,
        "first_dynamic_state_divergence_event": first_dynamic,
        "first_suppression_event": first_suppression,
        "first_divergence_equals_first_suppression": first_decision
        == first_suppression,
        "pre_divergence_identity": pre_divergence_identity,
        "divergence_proposal_equal": divergence_proposal_equal,
        "divergence_prestate_equal": divergence_prestate_equal,
        "divergence_decision_is_filter_only": divergence_decision_is_filter_only,
        "no_suppression_full_trace_equal": no_suppression_full_trace_equal,
        "control_stop_reason": control_row["stop_reason"],
        "filtered_stop_reason": filtered_row["stop_reason"],
        "control_completed": bool(control_row["completed"]),
        "filtered_completed": bool(filtered_row["completed"]),
        "delta_completed": int(filtered_row["completed"])
        - int(control_row["completed"]),
        "delta_spent_activations": filtered_row["cost_activations"]
        - control_row["cost_activations"],
        "delta_spent_accepted_swaps": filtered_row["cost_acceptedSwaps"]
        - control_row["cost_acceptedSwaps"],
        "delta_spent_full_ledger_unit_cost": filtered_row[
            "full_ledger_unit_cost"
        ]
        - control_row["full_ledger_unit_cost"],
    }
    both_complete = bool(control_row["completed"] and filtered_row["completed"])
    result["efficiency_comparable"] = both_complete
    result["efficiency_delta_activations"] = (
        result["delta_spent_activations"] if both_complete else None
    )
    result["efficiency_delta_accepted_swaps"] = (
        result["delta_spent_accepted_swaps"] if both_complete else None
    )
    result["efficiency_delta_full_ledger_unit_cost"] = (
        result["delta_spent_full_ledger_unit_cost"] if both_complete else None
    )
    for metric in METRICS:
        result[f"delta_peak_{metric}"] = (
            filtered_row[f"peak_{metric}"] - control_row[f"peak_{metric}"]
        )
        result[f"delta_excursion_{metric}"] = (
            filtered_row[f"excursion_{metric}"]
            - control_row[f"excursion_{metric}"]
        )
        result[f"delta_final_{metric}"] = (
            filtered_row[f"final_{metric}"] - control_row[f"final_{metric}"]
        )
    return result


def solve_filtered_graph(
    node_count: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    decision_codes: Sequence[int] | np.ndarray,
    metric_deltas: Sequence[int] | np.ndarray,
    metric_levels_: Sequence[int] | np.ndarray,
    terminal_codes: Sequence[int] | np.ndarray,
    threshold: int,
) -> tuple[Any, np.ndarray]:
    """Solve the exact graph after replacing suppressed changes by self-loops.

    Suppressed self-loops cannot create reachability, reduce a path maximum, or
    improve a positive-cost witness.  They are therefore omitted from the
    reachability/minimax solve while their count is retained for auditing.
    """

    source = np.asarray(sources, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    decisions = np.asarray(decision_codes, dtype=np.int8)
    deltas = np.asarray(metric_deltas, dtype=np.int64)
    suppressed = (decisions == 1) & (deltas > threshold)
    kept = ~suppressed
    solution = solve_primary_all_starts(
        node_count,
        source[kept],
        target[kept],
        metric_levels_,
        terminal_codes,
    )
    return solution, suppressed


def execute_extended_suppression_summary(
    family: FamilySpec,
    state_ordinal: int,
    *,
    source_family_ordinal: int,
    replicate_index: int,
    arm_variant: str,
    coupling_key: str,
    coupling_seed: int,
    metric: str,
    threshold: int,
    maximum_activations: int,
    prefix_activation: int,
) -> dict[str, Any]:
    """Summary-only long-horizon replay with an exact primary-prefix capture."""

    structural = structural_state_from_ordinal(family, state_ordinal)
    state = structural.to_run_state()
    scenario = make_scenario(
        family,
        seed=coupling_seed,
        max_activations=maximum_activations,
        generation_key=(
            f"E03-S09:{source_family_ordinal}:{state_ordinal}:"
            f"{replicate_index}:{arm_variant}"
        ),
    )
    state.terminal = evaluate_terminal(scenario, state)
    starts = metric_levels(scenario, state)
    peaks = dict(starts)
    filter_ = MetricWorseningFilter(metric, threshold)
    schedule = None
    if family.architecture == Architecture.CELL_VIEW:
        schedule = _CoupledSchedule(
            scenario,
            coupling_key=coupling_key,
            seed=coupling_seed,
            pulse_focal_id=None,
        )
    prefix: dict[str, Any] | None = None
    while state.terminal is None:
        execute_batch(
            scenario,
            state,
            retain_events=False,
            emit_event_records=False,
            schedule_factory=schedule,
            proposal_validation_filter=filter_,
        )
        levels = metric_levels(scenario, state)
        for name in METRICS:
            peaks[name] = max(peaks[name], levels[name])
        if state.activation_count == prefix_activation:
            prefix = {
                "prefix_dynamic_state_sha256": dynamic_state_digest(state),
                "prefix_ledger": dict(state.ledger),
                "prefix_levels": dict(levels),
                "prefix_stream_counters": dict(state.stream_counters),
                "prefix_filter_audit": filter_.audit_row(),
            }
    levels = metric_levels(scenario, state)
    final_structural = StructuralState.from_run_state(family, state)
    row: dict[str, Any] = {
        "extended_stop_reason": state.terminal,
        "extended_completed": state.terminal == "complete",
        "extended_event_budget_censored": state.terminal == "event_budget",
        "extended_event_count": state.activation_count,
        "extended_final_state_ordinal": final_structural.selection_cursor_code
        * math.factorial(family.n)
        + final_structural.occupancy_rank,
        "extended_final_dynamic_state_sha256": dynamic_state_digest(state),
        "extended_full_ledger_unit_cost": sum(state.ledger.values()),
        "extended_ledger_identities_valid": all(ledger_identity(state.ledger).values()),
        "prefix": prefix,
        "extended_filter_audit": filter_.audit_row(),
    }
    for key, value in state.ledger.items():
        row[f"extended_cost_{key}"] = value
    for name in METRICS:
        row[f"start_{name}"] = starts[name]
        row[f"extended_peak_{name}"] = peaks[name]
        row[f"extended_excursion_{name}"] = peaks[name] - starts[name]
        row[f"extended_final_{name}"] = levels[name]
    return row


def execute_suppression_arm_summary(
    family: FamilySpec,
    state_ordinal: int,
    *,
    source_family_ordinal: int,
    arm_family_ordinal: int,
    intervention_type: str,
    arm_variant: str,
    focal_index: int,
    target_index: int | None,
    replicate_index: int,
    coupling_key: str,
    coupling_seed: int,
    seed_search_attempts: int,
    maximum_activations: int,
    metric: str,
    threshold: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    MetricWorseningFilter,
    list[tuple[int, str]],
    list[tuple[int, int | None]],
    list[tuple[int, bool]],
]:
    """Run one filtered S10 arm with native records through first divergence.

    Before suppression, native event records are required for the isolation
    audit.  Once the filter has caused the declared divergence, the same engine
    continues in summary mode: proposals, validation, filter, commit, cost,
    terminal evaluation, counter streams, and metrics are unchanged, while
    redundant post-divergence event JSON/state hashing is skipped.  If no
    proposal is suppressed, the complete native trace is retained.
    """

    structural = structural_state_from_ordinal(family, state_ordinal)
    state = structural.to_run_state()
    scenario = make_scenario(
        family,
        seed=coupling_seed,
        max_activations=maximum_activations,
        generation_key=(
            f"E03-S09:{source_family_ordinal}:{state_ordinal}:"
            f"{replicate_index}:{arm_variant}"
        ),
    )
    state.terminal = evaluate_terminal(scenario, state)
    initial_terminal = state.terminal or "active"
    starts = metric_levels(scenario, state)
    levels = dict(starts)
    peaks = dict(starts)
    worsening = {name: 0 for name in METRICS}
    filter_ = MetricWorseningFilter(metric, threshold)
    schedule = None
    if family.architecture == Architecture.CELL_VIEW:
        schedule = _CoupledSchedule(
            scenario,
            coupling_key=coupling_key,
            seed=coupling_seed,
            pulse_focal_id=None,
        )
    retained_prefix: list[dict[str, Any]] = []
    actor_tokens: list[tuple[int, str]] = []
    side_tokens: list[tuple[int, int | None]] = []
    side_consumption: list[tuple[int, bool]] = []
    native_prefix_digest = bytes.fromhex(EMPTY_DIGEST)
    while state.terminal is None:
        event_index = state.activation_count
        before_dynamic = dynamic_state_digest(state)
        before_occupancy = tuple(state.occupancy)
        before_levels = dict(levels)
        before_swaps = state.ledger["acceptedSwaps"]
        token = (
            coupled_token(coupling_seed, coupling_key, event_index, family.n)
            if family.architecture == Architecture.CELL_VIEW
            else None
        )
        actor_tokens.append(
            (event_index, token.actor_id if token is not None else "controller")
        )
        side_tokens.append(
            (event_index, token.raw_side_value if token is not None else None)
        )
        emit_native = filter_.suppressed == 0
        events, encoded = execute_batch(
            scenario,
            state,
            retain_events=emit_native,
            emit_event_records=emit_native,
            schedule_factory=schedule,
            proposal_validation_filter=filter_,
        )
        if state.ledger["acceptedSwaps"] != before_swaps:
            levels = metric_levels(scenario, state)
            for name in METRICS:
                peaks[name] = max(peaks[name], levels[name])
                worsening[name] += int(levels[name] > before_levels[name])
        if emit_native:
            if len(events) != 1 or len(encoded) != 1:
                raise AssertionError("S10 summary runner requires serial native prefix")
            event = events[0]
            native_prefix_digest = hashlib.sha256(
                native_prefix_digest + encoded[0]
            ).digest()
            consumed = any(
                item["stream"] == "bubble_side"
                for item in event["randomAddressesAndDraws"]
            )
            side_consumption.append((event_index, consumed))
            retained_prefix.append(
                {
                    "event_index": event_index,
                    "actor_id": event["actorId"],
                    "raw_side": token.raw_side if token else None,
                    "raw_side_value": token.raw_side_value if token else None,
                    "side_consumed": consumed,
                    "proposal_kind": event["proposal"]["kind"],
                    "proposal_reason": event["proposal"]["reason"],
                    "proposal_actor_pos": event["proposal"]["actorPos"],
                    "proposal_target_pos": event["proposal"]["targetPos"],
                    "proposal_new_cursor": event["proposal"]["newCursor"],
                    "proposal_observation_reads": event["observation"]["reads"],
                    "proposal_value_comparisons": event["observation"][
                        "valueComparisons"
                    ],
                    "decision": event["decision"],
                    "pre_state_hash": event["preStateHash"],
                    "post_state_hash": event["postStateHash"],
                    "before_dynamic_hash": before_dynamic,
                    "after_dynamic_hash": dynamic_state_digest(state),
                    "before_occupancy": before_occupancy,
                    "after_occupancy": tuple(state.occupancy),
                    "before_levels": before_levels,
                    "after_levels": dict(levels),
                    "ledger_delta": dict(event["ledgerDelta"]),
                    "pulse_delivered": False,
                    "terminal": state.terminal,
                }
            )
        else:
            actor_index = token.actor_index if token is not None else -1
            consumed = (
                token is not None
                and family.policies[actor_index].value == "Bubble"
                and family.faults[actor_index].value == "normal"
            )
            side_consumption.append((event_index, consumed))

    final_structural = StructuralState.from_run_state(family, state)
    audit = filter_.audit_row()
    summary_payload = {
        "scenarioId": scenario.scenario_id,
        "state": state.to_dict(),
        "starts": starts,
        "peaks": peaks,
        "worsening": worsening,
        "filterAudit": audit,
        "actorTokens": actor_tokens,
        "sideTokens": side_tokens,
        "sideConsumption": side_consumption,
    }
    run_id = "s10summary:" + canonical_digest(
        {
            "sourceFamilyOrdinal": source_family_ordinal,
            "stateOrdinal": state_ordinal,
            "replicate": replicate_index,
            "armVariant": arm_variant,
            "metric": metric,
            "threshold": threshold,
        }
    )
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "research_step_id": "S10",
        "run_id": run_id,
        "pair_block_id": f"{source_family_ordinal}:{state_ordinal}:{replicate_index}",
        "source_family_ordinal": source_family_ordinal,
        "source_state_ordinal": state_ordinal,
        "arm_family_ordinal": arm_family_ordinal,
        "arm_state_ordinal": state_ordinal,
        "n": family.n,
        "architecture": family.architecture.value,
        "direction": family.direction.value,
        "policy_profile": family.policy_profile,
        "intervention_type": intervention_type,
        "arm_variant": arm_variant,
        "focal_barrier_id": f"c{focal_index}",
        "target_cell_id": f"c{target_index}" if target_index is not None else None,
        "replicate_index": replicate_index,
        "coupling_profile": "sha256_counter_E01_v1_common_pair_key_conditioned_focal_event0"
        if family.architecture == Architecture.CELL_VIEW
        else "not_applicable_traditional_controller",
        "coupling_key": coupling_key,
        "coupling_seed": str(coupling_seed),
        "seed_search_attempts": seed_search_attempts,
        "event_budget": maximum_activations,
        "pre_dynamic_state_sha256": dynamic_state_digest(structural.to_run_state()),
        "scenario_id": scenario.scenario_id,
        "initial_terminal": initial_terminal,
        "stop_reason": state.terminal,
        "completed": state.terminal == "complete",
        "event_budget_censored": state.terminal == "event_budget",
        "event_count": state.activation_count,
        "accepted_action_count": state.ledger["acceptedSwaps"]
        + state.ledger["memoryUpdates"],
        "pulse_delivered": None,
        "final_state_ordinal": final_structural.selection_cursor_code
        * math.factorial(family.n)
        + final_structural.occupancy_rank,
        "final_dynamic_state_sha256": dynamic_state_digest(state),
        "event_digest_sha256": None,
        "native_event_prefix_sha256": native_prefix_digest.hex(),
        "native_event_prefix_count": len(retained_prefix),
        "compact_trace_sha256": canonical_digest(retained_prefix),
        "run_summary_sha256": canonical_digest(summary_payload),
        "actor_token_prefix_sha256": canonical_digest(actor_tokens),
        "side_token_prefix_sha256": canonical_digest(side_tokens),
        "side_consumption_prefix_sha256": canonical_digest(side_consumption),
        "trace_complete": len(retained_prefix) == state.activation_count,
        "summary_complete": True,
        "ledger_identities_valid": all(ledger_identity(state.ledger).values()),
        "full_ledger_unit_cost": sum(state.ledger.values()),
    }
    for key, value in state.ledger.items():
        row[f"cost_{key}"] = value
    for name in METRICS:
        row[f"start_{name}"] = starts[name]
        row[f"peak_{name}"] = peaks[name]
        row[f"excursion_{name}"] = peaks[name] - starts[name]
        row[f"final_{name}"] = levels[name]
        row[f"worsening_events_{name}"] = worsening[name]
    return (
        row,
        retained_prefix,
        filter_,
        actor_tokens,
        side_tokens,
        side_consumption,
    )


def filtered_class_label(code: int) -> str:
    return CLASS_LABELS[int(code)]
