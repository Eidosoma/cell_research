"""Symmetry-breaking benchmark helpers for E05 S10."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.e05.cell_identity import identity_from_substrate_cell
from src.e05.scaling import (
    execute_swap_wait_fast,
    propose_scaling_action,
    target_identity_distance_lookup,
)
from src.e05.scrambled_embryo import (
    RUNNABLE_POLICY_IDS,
    clone_substrate_state,
    compute_state_metrics,
    policy_by_id,
    recovery_fraction,
    scramble_displacement_fraction,
    stable_int_hash,
)
from src.e05.substrates import SubstrateCell, SubstrateState
from src.e05.targets import TargetMorphology, symmetry_target


AXES = ("vertical", "horizontal")
AXIS_CHOICES = ("vertical", "horizontal", "ambiguous")
DEFAULT_SYMMETRY_SEEDS = (4101, 4102, 4103, 4104, 4105)
DEFAULT_EVENT_MULTIPLIER = 50
DEFAULT_RECORDS_PER_RUN = 25
DEFAULT_NEAR_SYMMETRY_BREAK_SWAPS = 2
AXIS_AMBIGUITY_TOLERANCE = 0.02


@dataclass(frozen=True)
class SymmetryInitialCondition:
    """One exact or near mirror-symmetric S10 initial-state condition."""

    condition_id: str
    construction_axis: str
    symmetry_class: str
    near_symmetry_break_swaps: int
    description: str

    def __post_init__(self) -> None:
        if self.construction_axis not in AXES:
            raise ValueError(f"unsupported construction_axis: {self.construction_axis}")
        if self.symmetry_class not in {"exact_axis_mirror", "near_axis_mirror"}:
            raise ValueError(f"unsupported symmetry_class: {self.symmetry_class}")
        if int(self.near_symmetry_break_swaps) < 0:
            raise ValueError("near_symmetry_break_swaps must be nonnegative")

    def compact_dict(self) -> dict[str, Any]:
        return {
            "research_step_id": "S10",
            "condition_id": self.condition_id,
            "construction_axis": self.construction_axis,
            "symmetry_class": self.symmetry_class,
            "near_symmetry_break_swaps": int(self.near_symmetry_break_swaps),
            "description": self.description,
            "organizer_cue_added": False,
        }


@dataclass(frozen=True)
class SymmetryRunResult:
    """One S10 symmetry-breaking trajectory and summary."""

    summary_row: dict[str, Any]
    trace_rows: tuple[dict[str, Any], ...]
    metric_rows: tuple[dict[str, Any], ...]
    initial_state: SubstrateState
    final_state: SubstrateState


def default_symmetry_target() -> TargetMorphology:
    """Return the square S03 bilateral-symmetry target used for axis choice."""

    return symmetry_target(7, 7, target_id="s10_bilateral_symmetry_square")


def default_initial_conditions() -> tuple[SymmetryInitialCondition, ...]:
    """Return exact and near symmetric initial-state conditions."""

    return (
        SymmetryInitialCondition(
            "exact_vertical_mirror",
            "vertical",
            "exact_axis_mirror",
            0,
            "Organ-label mirror symmetry is exact about the canonical vertical target axis.",
        ),
        SymmetryInitialCondition(
            "near_vertical_mirror",
            "vertical",
            "near_axis_mirror",
            DEFAULT_NEAR_SYMMETRY_BREAK_SWAPS,
            "Vertical mirror symmetry is deliberately perturbed by a small number of random cross-label swaps.",
        ),
        SymmetryInitialCondition(
            "exact_horizontal_mirror",
            "horizontal",
            "exact_axis_mirror",
            0,
            "Organ-label mirror symmetry is exact about the horizontal alternative axis.",
        ),
        SymmetryInitialCondition(
            "near_horizontal_mirror",
            "horizontal",
            "near_axis_mirror",
            DEFAULT_NEAR_SYMMETRY_BREAK_SWAPS,
            "Horizontal mirror symmetry is deliberately perturbed by a small number of random cross-label swaps.",
        ),
    )


def initial_condition_rows(
    target: TargetMorphology | None = None,
    conditions: Sequence[SymmetryInitialCondition] | None = None,
    *,
    seed: int = DEFAULT_SYMMETRY_SEEDS[0],
) -> list[dict[str, Any]]:
    """Return initial-condition rows with construction validation metrics."""

    target = target or default_symmetry_target()
    rows = []
    for condition in conditions or default_initial_conditions():
        state = construct_axis_symmetric_initial_state(target, condition, seed)
        metrics = axis_choice_metrics(target, state, "initial")
        row = condition.compact_dict()
        row.update(
            {
                "target_id": target.target_id,
                "target_kind": target.target_kind,
                "site_count": len(target.substrate.site_ids),
                "validation_seed": int(seed),
                "initial_target_error": float(target.target_error(state)),
                "construction_axis_label_symmetry_error": float(metrics[f"{condition.construction_axis}_label_symmetry_error"]),
                "vertical_label_symmetry_error": float(metrics["vertical_label_symmetry_error"]),
                "horizontal_label_symmetry_error": float(metrics["horizontal_label_symmetry_error"]),
                "initial_chosen_axis": metrics["chosen_axis"],
                "initial_axis_margin": float(metrics["axis_margin"]),
            }
        )
        rows.append(row)
    return rows


def construct_axis_symmetric_initial_state(
    target: TargetMorphology,
    condition: SymmetryInitialCondition,
    seed: int,
) -> SubstrateState:
    """Construct an exact or near mirror-symmetric random initial state."""

    rng = random.Random(int(seed) + stable_int_hash(condition.condition_id))
    state = _oriented_target_state(target, condition.construction_axis)
    state = _shuffle_axis_pairs_preserving_labels(target, state, condition.construction_axis, rng)
    if target.target_error(state) == 0.0:
        state = _shuffle_axis_pairs_preserving_labels(target, state, condition.construction_axis, rng)
    if condition.near_symmetry_break_swaps:
        state = _apply_near_symmetry_breaks(target, state, condition.construction_axis, condition.near_symmetry_break_swaps, rng)
    return state


def run_symmetry_breaking_benchmark(
    *,
    target: TargetMorphology | None = None,
    conditions: Sequence[SymmetryInitialCondition] | None = None,
    policy_ids: Sequence[str] = RUNNABLE_POLICY_IDS,
    seeds: Sequence[int] = DEFAULT_SYMMETRY_SEEDS,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, int], SymmetryRunResult]]:
    """Run the complete compact S10 benchmark matrix."""

    target = target or default_symmetry_target()
    conditions = tuple(conditions or default_initial_conditions())
    distance_lookup = target_identity_distance_lookup(target)
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    run_results: dict[tuple[str, str, int], SymmetryRunResult] = {}
    for condition in conditions:
        for policy_id in policy_ids:
            for seed in seeds:
                result = simulate_symmetry_breaking_run(
                    target=target,
                    condition=condition,
                    policy_id=policy_id,
                    seed=int(seed),
                    event_multiplier=event_multiplier,
                    records_per_run=records_per_run,
                    distance_lookup=distance_lookup,
                )
                summary_rows.append(result.summary_row)
                trace_rows.extend(result.trace_rows)
                metric_rows.extend(result.metric_rows)
                run_results[(condition.condition_id, policy_id, int(seed))] = result
    return summary_rows, trace_rows, metric_rows, run_results


def simulate_symmetry_breaking_run(
    *,
    target: TargetMorphology,
    condition: SymmetryInitialCondition,
    policy_id: str,
    seed: int,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
    distance_lookup: Mapping[tuple[str, int], float] | None = None,
) -> SymmetryRunResult:
    """Run one S10 trajectory from a symmetric or near-symmetric initial state."""

    policy = policy_by_id(policy_id)
    distance_lookup = distance_lookup or target_identity_distance_lookup(target)
    initial_state = construct_axis_symmetric_initial_state(target, condition, seed)
    state = clone_substrate_state(initial_state)
    site_ids = tuple(target.substrate.site_ids)
    event_cap = max(1, int(event_multiplier) * len(site_ids))
    record_every = max(1, event_cap // max(1, int(records_per_run)))
    rng = random.Random(int(seed) * 1_048_583 + stable_int_hash(f"S10:{condition.condition_id}:{policy_id}:{event_multiplier}"))
    initial_metrics = compute_state_metrics(target, state, "initial")
    initial_axis_metrics = axis_choice_metrics(target, state, "initial")
    trace_rows = [
        _trace_row(
            target=target,
            condition=condition,
            policy_id=policy.policy_id,
            policy_family=policy.policy_family,
            seed=seed,
            event_step=0,
            accepted_swaps=0,
            attempted_swaps=0,
            rejected_actions=0,
            wait_actions=0,
            energy_cost=0.0,
            target_error=initial_metrics["target_error"],
            aggregate_morphospace_error=initial_metrics["aggregate_morphospace_error"],
            axis_metrics=initial_axis_metrics,
        )
    ]
    metric_rows = _tag_metric_rows(
        initial_metrics["metric_rows"],
        target=target,
        condition=condition,
        policy_id=policy.policy_id,
        policy_family=policy.policy_family,
        seed=seed,
        state_label="initial",
    )

    accepted_swaps = 0
    attempted_swaps = 0
    rejected_actions = 0
    wait_actions = 0
    total_energy = 0.0
    population_initial = sum(1 for site_id in site_ids if state.cell_at(site_id) is not None)
    population_min = population_initial
    population_max = population_initial
    occupied_site_ids = tuple(site_id for site_id in site_ids if state.cell_at(site_id) is not None)
    final_axis_metrics = initial_axis_metrics
    final_metrics = initial_metrics
    for event_step in range(1, event_cap + 1):
        if not occupied_site_ids:
            break
        actor_site_id = rng.choice(occupied_site_ids)
        request = propose_scaling_action(
            policy_id=policy_id,
            state=state,
            target=target,
            actor_site_id=actor_site_id,
            rng=rng,
            distance_lookup=distance_lookup,
        )
        result = execute_swap_wait_fast(state, request, population_current=population_initial)
        if request.action_type == "swap":
            attempted_swaps += 1
        if request.action_type == "wait":
            wait_actions += 1
        if not result.allowed:
            rejected_actions += 1
        if result.allowed and result.action_type == "swap" and result.state_changed:
            accepted_swaps += 1
        total_energy += float(result.energy_cost_charged)
        population_min = min(population_min, int(result.population_after))
        population_max = max(population_max, int(result.population_after))
        if event_step % record_every == 0 or event_step == event_cap:
            final_metrics = compute_state_metrics(target, state, f"event_{event_step}")
            final_axis_metrics = axis_choice_metrics(target, state, f"event_{event_step}")
            trace_rows.append(
                _trace_row(
                    target=target,
                    condition=condition,
                    policy_id=policy.policy_id,
                    policy_family=policy.policy_family,
                    seed=seed,
                    event_step=event_step,
                    accepted_swaps=accepted_swaps,
                    attempted_swaps=attempted_swaps,
                    rejected_actions=rejected_actions,
                    wait_actions=wait_actions,
                    energy_cost=total_energy,
                    target_error=final_metrics["target_error"],
                    aggregate_morphospace_error=final_metrics["aggregate_morphospace_error"],
                    axis_metrics=final_axis_metrics,
                )
            )
    final_metrics = compute_state_metrics(target, state, "final")
    final_axis_metrics = axis_choice_metrics(target, state, "final")
    metric_rows.extend(
        _tag_metric_rows(
            final_metrics["metric_rows"],
            target=target,
            condition=condition,
            policy_id=policy.policy_id,
            policy_family=policy.policy_family,
            seed=seed,
            state_label="final",
        )
    )
    if trace_rows[-1]["event_step"] != event_cap:
        trace_rows.append(
            _trace_row(
                target=target,
                condition=condition,
                policy_id=policy.policy_id,
                policy_family=policy.policy_family,
                seed=seed,
                event_step=event_cap,
                accepted_swaps=accepted_swaps,
                attempted_swaps=attempted_swaps,
                rejected_actions=rejected_actions,
                wait_actions=wait_actions,
                energy_cost=total_energy,
                target_error=final_metrics["target_error"],
                aggregate_morphospace_error=final_metrics["aggregate_morphospace_error"],
                axis_metrics=final_axis_metrics,
            )
        )
    else:
        trace_rows[-1]["target_error"] = final_metrics["target_error"]
        trace_rows[-1]["aggregate_morphospace_error"] = final_metrics["aggregate_morphospace_error"]
        _update_trace_axis_metrics(trace_rows[-1], final_axis_metrics)

    final_population = sum(1 for site_id in site_ids if state.cell_at(site_id) is not None)
    axis_switch_count = _axis_switch_count(trace_rows)
    ambiguous_fraction = _ambiguous_step_fraction(trace_rows)
    final_axis_stable = _final_axis_stable(trace_rows)
    initial_target_error = float(initial_metrics["target_error"])
    final_target_error = float(final_metrics["target_error"])
    initial_aggregate = float(initial_metrics["aggregate_morphospace_error"])
    final_aggregate = float(final_metrics["aggregate_morphospace_error"])
    summary_row = {
        "research_step_id": "S10",
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "target_hash": target.target_hash,
        "substrate_kind": target.substrate.kind,
        "site_count": len(site_ids),
        "condition_id": condition.condition_id,
        "construction_axis": condition.construction_axis,
        "symmetry_class": condition.symmetry_class,
        "near_symmetry_break_swaps": int(condition.near_symmetry_break_swaps),
        "policy_id": policy.policy_id,
        "policy_family": policy.policy_family,
        "declared_information_access": policy.declared_information_access,
        "policy_parameter_signature": policy_parameter_signature(policy_id, event_multiplier),
        "organizer_cue_added": False,
        "simulation_seed": int(seed),
        "event_multiplier": int(event_multiplier),
        "records_per_run": int(records_per_run),
        "event_cap": int(event_cap),
        "record_every": int(record_every),
        "initial_target_error": initial_target_error,
        "final_target_error": final_target_error,
        "delta_target_error": final_target_error - initial_target_error,
        "target_recovery_fraction": recovery_fraction(initial_target_error, final_target_error),
        "initial_aggregate_morphospace_error": initial_aggregate,
        "final_aggregate_morphospace_error": final_aggregate,
        "delta_aggregate_morphospace_error": final_aggregate - initial_aggregate,
        "aggregate_recovery_fraction": recovery_fraction(initial_aggregate, final_aggregate),
        "initial_displacement_fraction": scramble_displacement_fraction(target, initial_state),
        "final_displacement_fraction": scramble_displacement_fraction(target, state),
        "initial_chosen_axis": initial_axis_metrics["chosen_axis"],
        "final_chosen_axis": final_axis_metrics["chosen_axis"],
        "initial_axis_margin": float(initial_axis_metrics["axis_margin"]),
        "final_axis_margin": float(final_axis_metrics["axis_margin"]),
        "initial_vertical_orientation_error": float(initial_axis_metrics["vertical_orientation_error"]),
        "initial_horizontal_orientation_error": float(initial_axis_metrics["horizontal_orientation_error"]),
        "final_vertical_orientation_error": float(final_axis_metrics["vertical_orientation_error"]),
        "final_horizontal_orientation_error": float(final_axis_metrics["horizontal_orientation_error"]),
        "initial_vertical_label_symmetry_error": float(initial_axis_metrics["vertical_label_symmetry_error"]),
        "initial_horizontal_label_symmetry_error": float(initial_axis_metrics["horizontal_label_symmetry_error"]),
        "final_vertical_label_symmetry_error": float(final_axis_metrics["vertical_label_symmetry_error"]),
        "final_horizontal_label_symmetry_error": float(final_axis_metrics["horizontal_label_symmetry_error"]),
        "construction_axis_label_symmetry_error_initial": float(initial_axis_metrics[f"{condition.construction_axis}_label_symmetry_error"]),
        "construction_axis_label_symmetry_error_final": float(final_axis_metrics[f"{condition.construction_axis}_label_symmetry_error"]),
        "canonical_axis_selected_initial": initial_axis_metrics["chosen_axis"] == "vertical",
        "canonical_axis_selected_final": final_axis_metrics["chosen_axis"] == "vertical",
        "construction_axis_retained_final": final_axis_metrics["chosen_axis"] == condition.construction_axis,
        "axis_switched_to_canonical": condition.construction_axis != "vertical" and final_axis_metrics["chosen_axis"] == "vertical",
        "axis_choice_outcome": _axis_choice_outcome(condition, final_axis_metrics["chosen_axis"]),
        "axis_switch_count": int(axis_switch_count),
        "ambiguous_step_fraction": float(ambiguous_fraction),
        "final_axis_stable_last_quarter": bool(final_axis_stable),
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "wait_actions": int(wait_actions),
        "total_energy_cost": float(total_energy),
        "energy_per_site": float(total_energy / max(1, len(site_ids))),
        "population_initial": int(population_initial),
        "population_final": int(final_population),
        "population_min": int(population_min),
        "population_max": int(population_max),
        "population_delta_total": int(final_population - population_initial),
    }
    return SymmetryRunResult(
        summary_row=summary_row,
        trace_rows=tuple(trace_rows),
        metric_rows=tuple(metric_rows),
        initial_state=initial_state,
        final_state=clone_substrate_state(state),
    )


def axis_choice_metrics(target: TargetMorphology, state: SubstrateState, state_label: str) -> dict[str, Any]:
    """Return axis-choice metrics without adding organizer cues."""

    vertical_orientation = axis_orientation_target_error(target, state, "vertical")
    horizontal_orientation = axis_orientation_target_error(target, state, "horizontal")
    margin = abs(horizontal_orientation - vertical_orientation)
    if margin <= AXIS_AMBIGUITY_TOLERANCE:
        chosen_axis = "ambiguous"
    elif vertical_orientation < horizontal_orientation:
        chosen_axis = "vertical"
    else:
        chosen_axis = "horizontal"
    return {
        "state_label": state_label,
        "vertical_orientation_error": float(vertical_orientation),
        "horizontal_orientation_error": float(horizontal_orientation),
        "vertical_label_symmetry_error": axis_label_symmetry_error(target, state, "vertical"),
        "horizontal_label_symmetry_error": axis_label_symmetry_error(target, state, "horizontal"),
        "axis_margin": float(margin),
        "chosen_axis": chosen_axis,
        "axis_ambiguity_tolerance": AXIS_AMBIGUITY_TOLERANCE,
    }


def axis_orientation_target_error(target: TargetMorphology, state: SubstrateState, axis: str) -> float:
    """Mean identity error to a vertical or horizontal orientation of the target."""

    _validate_axis(axis)
    total = 0.0
    for site_id in target.substrate.site_ids:
        cell = state.cell_at(site_id)
        if cell is None:
            total += 1.0
            continue
        identity = identity_from_substrate_cell(cell, fallback_scalar=False)
        source_site_id = _orientation_source_site_id(target, site_id, axis)
        total += float(target.schema.distance(identity, target.identities_by_site[source_site_id]))
    return float(total / len(target.substrate.site_ids))


def axis_label_symmetry_error(target: TargetMorphology, state: SubstrateState, axis: str) -> float:
    """Fraction of mirror pairs with different observed organ labels."""

    _validate_axis(axis)
    mismatches = 0
    pair_count = 0
    for pair in axis_site_pairs(target, axis):
        if len(pair) != 2:
            continue
        left_site_id, right_site_id = pair
        if left_site_id == right_site_id:
            continue
        pair_count += 1
        if _observed_organ_label(state, left_site_id) != _observed_organ_label(state, right_site_id):
            mismatches += 1
    return float(mismatches / pair_count) if pair_count else 0.0


def axis_site_pairs(target: TargetMorphology, axis: str) -> tuple[tuple[int, int], ...]:
    """Return mirror-pair site IDs for one candidate axis."""

    _validate_axis(axis)
    pairs = []
    visited: set[int] = set()
    for site_id in target.substrate.site_ids:
        if site_id in visited:
            continue
        mirror = _mirror_site_id(target, site_id, axis)
        if int(site_id) == int(mirror):
            pair = (int(site_id),)
        else:
            pair = (int(site_id), int(mirror)) if int(site_id) <= int(mirror) else (int(mirror), int(site_id))
        visited.update(pair)
        pairs.append(pair)
    return tuple(pairs)


def policy_parameter_signature(policy_id: str, event_multiplier: int) -> str:
    return json.dumps(
        {
            "policy_id": policy_id,
            "action_set": "S04 default swap/wait legality and energy semantics, optimized audit-free inner loop",
            "event_multiplier_per_site": int(event_multiplier),
            "policy_code_path": "src/e05/scrambled_embryo.py",
            "organizer_cue_added": False,
            "global_axis_observation_added": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def source_guardrail_rows() -> tuple[dict[str, Any], ...]:
    """Return policy information-access guardrail rows for S10."""

    rows = []
    for policy_id in RUNNABLE_POLICY_IDS:
        policy = policy_by_id(policy_id)
        rows.append(
            {
                "research_step_id": "S10",
                "policy_id": policy.policy_id,
                "policy_family": policy.policy_family,
                "declared_information_access": policy.declared_information_access,
                "organizer_cue_added": False,
                "global_axis_observation_added": False,
                "notes": "S10 reuses the S07 runnable control without expanding observation access or adding organizer cells.",
            }
        )
    return tuple(rows)


def _oriented_target_state(target: TargetMorphology, axis: str) -> SubstrateState:
    _validate_axis(axis)
    source = target.constructed_substrate()
    state = target.substrate.copy_empty()
    used: set[str] = set()
    for site_id in target.substrate.site_ids:
        source_site_id = _orientation_source_site_id(target, site_id, axis)
        cell = source.cell_at(source_site_id)
        if cell is None:
            raise ValueError("constructed target unexpectedly has empty cells")
        if str(cell.cell_id) in used:
            raise ValueError("orientation mapping reused a cell; square target required for horizontal axis")
        used.add(str(cell.cell_id))
        state.place_cell(site_id, cell)
    return state


def _shuffle_axis_pairs_preserving_labels(
    target: TargetMorphology,
    state: SubstrateState,
    axis: str,
    rng: random.Random,
) -> SubstrateState:
    groups: dict[tuple[str, int], list[tuple[tuple[int, int], tuple[SubstrateCell, ...]]]] = {}
    for pair in axis_site_pairs(target, axis):
        cells = tuple(state.cell_at(site_id) for site_id in pair)
        if any(cell is None for cell in cells):
            raise ValueError("axis-pair shuffle requires occupied sites")
        labels = tuple(sorted(_identity_organ_label(cell) for cell in cells if cell is not None))
        groups.setdefault(("|".join(labels), len(pair)), []).append((pair, cells))  # type: ignore[arg-type]
    output = target.substrate.copy_empty()
    for items in groups.values():
        payloads = [cells for _pair, cells in items]
        rng.shuffle(payloads)
        for (pair, _old_cells), cells in zip(items, payloads, strict=True):
            placed = tuple(cells)
            if len(pair) == 2 and pair[0] != pair[1] and rng.random() < 0.5:
                placed = tuple(reversed(placed))
            for site_id, cell in zip(pair, placed, strict=True):
                output.place_cell(site_id, cell)
    return output


def _apply_near_symmetry_breaks(
    target: TargetMorphology,
    state: SubstrateState,
    construction_axis: str,
    swap_count: int,
    rng: random.Random,
) -> SubstrateState:
    output = clone_substrate_state(state)
    site_ids = list(target.substrate.site_ids)
    for _ in range(int(swap_count)):
        candidates = [
            (left, right)
            for left in site_ids
            for right in site_ids
            if left < right
            and _observed_organ_label(output, left) != _observed_organ_label(output, right)
            and _mirror_site_id(target, left, construction_axis) != right
        ]
        if not candidates:
            break
        left, right = rng.choice(candidates)
        output._swap_unchecked(left, right)
    return output


def _orientation_source_site_id(target: TargetMorphology, site_id: int, axis: str) -> int:
    _validate_axis(axis)
    if axis == "vertical":
        return int(site_id)
    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    height = int(target.substrate.dimensions.get("height", 1))
    if width != height:
        raise ValueError("horizontal orientation target comparison requires a square target")
    x, y = target.substrate.coordinate(site_id)
    return int(x * width + y)


def _mirror_site_id(target: TargetMorphology, site_id: int, axis: str) -> int:
    _validate_axis(axis)
    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    height = int(target.substrate.dimensions.get("height", 1))
    x, y = target.substrate.coordinate(site_id)
    if axis == "vertical":
        return int(y * width + (width - 1 - x))
    return int((height - 1 - y) * width + x)


def _observed_organ_label(state: SubstrateState, site_id: int) -> str:
    cell = state.cell_at(site_id)
    if cell is None:
        return "__empty__"
    return _identity_organ_label(cell)


def _identity_organ_label(cell: SubstrateCell) -> str:
    try:
        identity = identity_from_substrate_cell(cell, fallback_scalar=False)
    except ValueError:
        return str(cell.label)
    return str(identity.components.get("organ_type", cell.label))


def _tag_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: TargetMorphology,
    condition: SymmetryInitialCondition,
    policy_id: str,
    policy_family: str,
    seed: int,
    state_label: str,
) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        payload = dict(row)
        payload.update(
            {
                "research_step_id": "S10",
                "target_kind": target.target_kind,
                "condition_id": condition.condition_id,
                "construction_axis": condition.construction_axis,
                "symmetry_class": condition.symmetry_class,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "simulation_seed": int(seed),
                "benchmark_state_label": state_label,
            }
        )
        tagged.append(payload)
    return tagged


def _trace_row(
    *,
    target: TargetMorphology,
    condition: SymmetryInitialCondition,
    policy_id: str,
    policy_family: str,
    seed: int,
    event_step: int,
    accepted_swaps: int,
    attempted_swaps: int,
    rejected_actions: int,
    wait_actions: int,
    energy_cost: float,
    target_error: float,
    aggregate_morphospace_error: float,
    axis_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "research_step_id": "S10",
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "condition_id": condition.condition_id,
        "construction_axis": condition.construction_axis,
        "symmetry_class": condition.symmetry_class,
        "policy_id": policy_id,
        "policy_family": policy_family,
        "simulation_seed": int(seed),
        "event_step": int(event_step),
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "wait_actions": int(wait_actions),
        "energy_cost": float(energy_cost),
        "target_error": float(target_error),
        "aggregate_morphospace_error": float(aggregate_morphospace_error),
    }
    _update_trace_axis_metrics(row, axis_metrics)
    return row


def _update_trace_axis_metrics(row: dict[str, Any], axis_metrics: Mapping[str, Any]) -> None:
    row.update(
        {
            "chosen_axis": axis_metrics["chosen_axis"],
            "axis_margin": float(axis_metrics["axis_margin"]),
            "vertical_orientation_error": float(axis_metrics["vertical_orientation_error"]),
            "horizontal_orientation_error": float(axis_metrics["horizontal_orientation_error"]),
            "vertical_label_symmetry_error": float(axis_metrics["vertical_label_symmetry_error"]),
            "horizontal_label_symmetry_error": float(axis_metrics["horizontal_label_symmetry_error"]),
        }
    )


def _axis_switch_count(trace_rows: Sequence[Mapping[str, Any]]) -> int:
    choices = [str(row["chosen_axis"]) for row in trace_rows]
    return sum(1 for left, right in zip(choices, choices[1:], strict=False) if left != right)


def _ambiguous_step_fraction(trace_rows: Sequence[Mapping[str, Any]]) -> float:
    if not trace_rows:
        return 0.0
    return float(sum(1 for row in trace_rows if row["chosen_axis"] == "ambiguous") / len(trace_rows))


def _final_axis_stable(trace_rows: Sequence[Mapping[str, Any]]) -> bool:
    if not trace_rows:
        return False
    final_choice = str(trace_rows[-1]["chosen_axis"])
    if final_choice == "ambiguous":
        return False
    window = max(1, len(trace_rows) // 4)
    return all(str(row["chosen_axis"]) == final_choice for row in trace_rows[-window:])


def _axis_choice_outcome(condition: SymmetryInitialCondition, final_axis: str) -> str:
    if final_axis == "ambiguous":
        return "ambiguous_or_unresolved"
    if final_axis == "vertical":
        return "canonical_vertical_selected"
    if final_axis == condition.construction_axis:
        return "construction_axis_retained"
    return "noncanonical_axis_selected"


def _validate_axis(axis: str) -> None:
    if axis not in AXES:
        raise ValueError(f"unsupported axis: {axis}")
