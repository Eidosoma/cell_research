"""Scaling benchmark helpers for E05 S09."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.e05.actions import MorphogenesisActionRequest
from src.e05.cell_identity import identity_from_substrate_cell
from src.e05.regeneration import PatchMask, semantic_blocker_for_perturbation
from src.e05.scrambled_embryo import (
    RUNNABLE_POLICY_IDS,
    clone_substrate_state,
    compute_state_metrics,
    policy_by_id,
    recovery_fraction,
    scramble_displacement_fraction,
    scramble_target_state,
    stable_int_hash,
)
from src.e05.substrates import SubstrateState
from src.e05.targets import (
    TargetMorphology,
    boundary_target,
    gradient_target,
    organ_like_target,
    ring_target,
    stripes_target,
    symmetry_target,
)


SCALING_TASK_TYPES = ("scrambled_embryo", "rotated_patch_regeneration")
DEFAULT_SCALING_SEEDS = (3101, 3102, 3103)
DEFAULT_EVENT_MULTIPLIER = 50
DEFAULT_RECORDS_PER_RUN = 20
LOW_RECOVERY_FAILURE_THRESHOLD = 0.05


@dataclass(frozen=True)
class TargetScaleConfig:
    """Small or large S03-compatible target configuration for S09."""

    config_id: str
    source_target_id: str
    target_kind: str
    constructor_name: str
    scale_label: str
    width: int
    height: int
    scale_factor: float
    scaled_from_width: int
    scaled_from_height: int
    proportion_rule: str
    target_id: str

    @property
    def site_count(self) -> int:
        return int(self.width) * int(self.height)

    @property
    def dimension_signature(self) -> str:
        return f"{self.width}x{self.height}"

    def make_target(self) -> TargetMorphology:
        constructor = _target_constructor(self.constructor_name)
        return constructor(self.width, self.height, target_id=self.target_id)

    def compact_dict(self) -> dict[str, Any]:
        return {
            "research_step_id": "S09",
            "config_id": self.config_id,
            "source_target_id": self.source_target_id,
            "target_id": self.target_id,
            "target_kind": self.target_kind,
            "constructor_name": self.constructor_name,
            "scale_label": self.scale_label,
            "width": int(self.width),
            "height": int(self.height),
            "site_count": self.site_count,
            "dimension_signature": self.dimension_signature,
            "scale_factor": float(self.scale_factor),
            "scaled_from_width": int(self.scaled_from_width),
            "scaled_from_height": int(self.scaled_from_height),
            "proportion_rule": self.proportion_rule,
            "retuning_allowed": False,
        }


@dataclass(frozen=True)
class ScaledPerturbation:
    """One proportionally scaled rotated-patch perturbation."""

    mask: PatchMask
    state: SubstrateState
    baseline_target_error: float
    perturbed_target_error: float
    perturbed_aggregate_morphospace_error: float
    population_baseline: int
    population_perturbed: int


@dataclass(frozen=True)
class ScalingRunResult:
    """One S09 size-transfer run."""

    summary_row: dict[str, Any]
    metric_rows: tuple[dict[str, Any], ...]
    initial_state: SubstrateState
    final_state: SubstrateState


@dataclass(frozen=True)
class FastSwapWaitOutcome:
    """Minimal S04-equivalent outcome for S09 swap/wait-only controls."""

    action_type: str
    allowed: bool
    reason: str
    state_changed: bool
    energy_cost_charged: float
    population_after: int


def default_scaling_target_configs() -> tuple[TargetScaleConfig, ...]:
    """Return paired small/large target configs for all genuine 2D S03 families."""

    specs = (
        ("ap_gradient", "gradient", "gradient_target", 6, 4, "normalized anterior-posterior coordinate over width"),
        ("organ_stripes", "stripes", "stripes_target", 6, 4, "alternating organ stripes along x using the same cell-level rule"),
        ("boundary_pattern", "boundary", "boundary_target", 6, 4, "fixed one-cell perimeter boundary on a larger rectangle"),
        ("ring_pattern", "ring", "ring_target", 7, 7, "radius proportional to min(width,height) with fixed discrete ring tolerance"),
        ("bilateral_symmetry", "symmetry", "symmetry_target", 7, 5, "central left-right axis with odd width preserved"),
        ("toy_organ_like", "organ_like", "organ_like_target", 8, 5, "fractional body-region rules from the S03 constructor"),
    )
    configs: list[TargetScaleConfig] = []
    for source_target_id, target_kind, constructor_name, width, height, rule in specs:
        for scale_label in ("small", "large"):
            scale_factor = 1.0 if scale_label == "small" else 2.0
            scaled_width = width if scale_label == "small" else _scaled_dimension(width, target_kind, axis="width")
            scaled_height = height if scale_label == "small" else _scaled_dimension(height, target_kind, axis="height")
            target_id = f"{source_target_id}_{scale_label}"
            configs.append(
                TargetScaleConfig(
                    config_id=f"{source_target_id}:{scale_label}",
                    source_target_id=source_target_id,
                    target_kind=target_kind,
                    constructor_name=constructor_name,
                    scale_label=scale_label,
                    width=int(scaled_width),
                    height=int(scaled_height),
                    scale_factor=scale_factor,
                    scaled_from_width=int(width),
                    scaled_from_height=int(height),
                    proportion_rule=rule,
                    target_id=target_id,
                )
            )
    return tuple(configs)


def target_config_rows(configs: Sequence[TargetScaleConfig] | None = None) -> list[dict[str, Any]]:
    """Return config rows with constructed-target validation fields."""

    rows: list[dict[str, Any]] = []
    small_by_source = {
        config.source_target_id: config
        for config in configs or default_scaling_target_configs()
        if config.scale_label == "small"
    }
    for config in configs or default_scaling_target_configs():
        target = config.make_target()
        small = small_by_source[config.source_target_id]
        row = config.compact_dict()
        row.update(
            {
                "substrate_kind": target.substrate.kind,
                "target_hash": target.target_hash,
                "constructed_target_error": float(target.target_error(target.constructed_substrate())),
                "large_to_small_site_ratio": float(config.site_count / small.site_count),
                "large_to_small_width_ratio": float(config.width / small.width),
                "large_to_small_height_ratio": float(config.height / small.height),
                "target_schema_id": target.schema.schema_id,
            }
        )
        rows.append(row)
    return rows


def validate_scaling_target_configs(configs: Sequence[TargetScaleConfig] | None = None) -> tuple[dict[str, Any], ...]:
    """Validate pair coverage and constructed-state zero errors."""

    configs = tuple(configs or default_scaling_target_configs())
    rows = target_config_rows(configs)
    by_source: dict[str, list[TargetScaleConfig]] = {}
    for config in configs:
        by_source.setdefault(config.source_target_id, []).append(config)
    pair_rows = []
    for source_target_id, pair in sorted(by_source.items()):
        labels = sorted(config.scale_label for config in pair)
        small = next(config for config in pair if config.scale_label == "small")
        large = next(config for config in pair if config.scale_label == "large")
        pair_rows.append(
            {
                "research_step_id": "S09",
                "validation_case": f"scaled_pair_{source_target_id}",
                "success": labels == ["large", "small"]
                and large.width > small.width
                and large.height > small.height
                and large.site_count > small.site_count,
                "source_target_id": source_target_id,
                "small_dimensions": small.dimension_signature,
                "large_dimensions": large.dimension_signature,
                "site_ratio": float(large.site_count / small.site_count),
                "detail": "Small and large paired configs exist and large has more sites in both dimensions.",
            }
        )
    zero_rows = [
        {
            "research_step_id": "S09",
            "validation_case": "constructed_scaled_target_zero_error",
            "success": all(float(row["constructed_target_error"]) == 0.0 for row in rows),
            "checked_target_count": len(rows),
            "max_constructed_target_error": max(float(row["constructed_target_error"]) for row in rows),
            "detail": "Every scaled target constructs a state with zero S03 target error.",
        }
    ]
    return tuple([*pair_rows, *zero_rows])


def run_scaling_benchmark(
    *,
    configs: Sequence[TargetScaleConfig] | None = None,
    task_types: Sequence[str] = SCALING_TASK_TYPES,
    policy_ids: Sequence[str] = RUNNABLE_POLICY_IDS,
    seeds: Sequence[int] = DEFAULT_SCALING_SEEDS,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str, int], ScalingRunResult]]:
    """Run the complete compact S09 size-transfer benchmark."""

    configs = tuple(configs or default_scaling_target_configs())
    _validate_task_types(task_types)
    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    run_results: dict[tuple[str, str, str, int], ScalingRunResult] = {}
    targets_by_id = {config.target_id: config.make_target() for config in configs}
    distance_lookup_by_id = {
        target_id: target_identity_distance_lookup(target)
        for target_id, target in targets_by_id.items()
    }
    for config in configs:
        target = targets_by_id[config.target_id]
        for task_type in task_types:
            for policy_id in policy_ids:
                for seed in seeds:
                    result = simulate_scaling_run(
                        config=config,
                        target=target,
                        task_type=task_type,
                        policy_id=policy_id,
                        seed=int(seed),
                        event_multiplier=event_multiplier,
                        records_per_run=records_per_run,
                        distance_lookup=distance_lookup_by_id[config.target_id],
                    )
                    summary_rows.append(result.summary_row)
                    metric_rows.extend(result.metric_rows)
                    run_results[(config.config_id, task_type, policy_id, int(seed))] = result
    return summary_rows, metric_rows, run_results


def simulate_scaling_run(
    *,
    config: TargetScaleConfig,
    target: TargetMorphology,
    task_type: str,
    policy_id: str,
    seed: int,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
    distance_lookup: Mapping[tuple[str, int], float] | None = None,
) -> ScalingRunResult:
    """Run one no-retuning size-transfer trajectory."""

    if task_type not in SCALING_TASK_TYPES:
        raise ValueError(f"unsupported S09 task_type: {task_type}")
    policy = policy_by_id(policy_id)
    initial_state, task_metadata = _initial_state_for_task(config=config, target=target, task_type=task_type, seed=seed)
    state = clone_substrate_state(initial_state)
    distance_lookup = distance_lookup or target_identity_distance_lookup(target)
    site_ids = tuple(target.substrate.site_ids)
    event_cap = max(1, int(event_multiplier) * len(site_ids))
    rng = random.Random(
        int(seed) * 524_287 + stable_int_hash(f"S09:{config.config_id}:{task_type}:{policy_id}:{event_multiplier}")
    )
    initial_metrics = compute_state_metrics(target, state, "initial")
    metric_rows = _tag_metric_rows(
        initial_metrics["metric_rows"],
        config=config,
        task_type=task_type,
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
    for _event_step in range(1, event_cap + 1):
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

    final_metrics = compute_state_metrics(target, state, "final")
    metric_rows.extend(
        _tag_metric_rows(
            final_metrics["metric_rows"],
            config=config,
            task_type=task_type,
            policy_id=policy.policy_id,
            policy_family=policy.policy_family,
            seed=seed,
            state_label="final",
        )
    )
    initial_target_error = float(initial_metrics["target_error"])
    final_target_error = float(final_metrics["target_error"])
    initial_aggregate = float(initial_metrics["aggregate_morphospace_error"])
    final_aggregate = float(final_metrics["aggregate_morphospace_error"])
    target_recovery = recovery_fraction(initial_target_error, final_target_error)
    aggregate_recovery = recovery_fraction(initial_aggregate, final_aggregate)
    final_population = sum(1 for site_id in site_ids if state.cell_at(site_id) is not None)
    summary_row = {
        "research_step_id": "S09",
        "task_type": task_type,
        "source_task_basis": "S07_scrambled_embryo" if task_type == "scrambled_embryo" else "S08_rotated_patch_repairable",
        "config_id": config.config_id,
        "source_target_id": config.source_target_id,
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "target_hash": target.target_hash,
        "substrate_kind": target.substrate.kind,
        "scale_label": config.scale_label,
        "scale_factor": float(config.scale_factor),
        "width": int(config.width),
        "height": int(config.height),
        "site_count": len(site_ids),
        "dimension_signature": config.dimension_signature,
        "proportion_rule": config.proportion_rule,
        "policy_id": policy.policy_id,
        "policy_family": policy.policy_family,
        "declared_information_access": policy.declared_information_access,
        "policy_parameter_signature": policy_parameter_signature(policy_id, event_multiplier),
        "retuning_allowed": False,
        "large_grid_retuned": False,
        "tuned_on_scale_label": "small",
        "simulation_seed": int(seed),
        "event_multiplier": int(event_multiplier),
        "records_per_run": int(records_per_run),
        "event_cap": int(event_cap),
        "initial_target_error": initial_target_error,
        "final_target_error": final_target_error,
        "delta_target_error": final_target_error - initial_target_error,
        "target_recovery_fraction": target_recovery,
        "initial_aggregate_morphospace_error": initial_aggregate,
        "final_aggregate_morphospace_error": final_aggregate,
        "delta_aggregate_morphospace_error": final_aggregate - initial_aggregate,
        "aggregate_recovery_fraction": aggregate_recovery,
        "initial_displacement_fraction": scramble_displacement_fraction(target, initial_state),
        "final_displacement_fraction": scramble_displacement_fraction(target, state),
        "failure_to_improve": bool(final_target_error >= initial_target_error - 1e-12),
        "low_recovery_failure": bool(target_recovery < LOW_RECOVERY_FAILURE_THRESHOLD),
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "wait_actions": int(wait_actions),
        "total_energy_cost": float(total_energy),
        "energy_per_site": float(total_energy / max(1, len(site_ids))),
        "accepted_swaps_per_site": float(accepted_swaps / max(1, len(site_ids))),
        "attempted_swaps_per_site": float(attempted_swaps / max(1, len(site_ids))),
        "population_initial": int(population_initial),
        "population_final": int(final_population),
        "population_min": int(population_min),
        "population_max": int(population_max),
        "population_delta_total": int(final_population - population_initial),
        **task_metadata,
    }
    return ScalingRunResult(
        summary_row=summary_row,
        metric_rows=tuple(metric_rows),
        initial_state=initial_state,
        final_state=clone_substrate_state(state),
    )


def scaling_summary_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate run-level rows into small-to-large scaling comparisons."""

    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in summary_rows:
        key = (
            str(row["task_type"]),
            str(row["target_kind"]),
            str(row["policy_id"]),
            str(row["scale_label"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (task_type, target_kind, policy_id, scale_label), rows in sorted(groups.items()):
        output.append(
            {
                "research_step_id": "S09",
                "task_type": task_type,
                "target_kind": target_kind,
                "policy_id": policy_id,
                "scale_label": scale_label,
                "runs": len(rows),
                "site_count": int(rows[0]["site_count"]),
                "mean_initial_target_error": _mean(row["initial_target_error"] for row in rows),
                "mean_final_target_error": _mean(row["final_target_error"] for row in rows),
                "mean_target_recovery_fraction": _mean(row["target_recovery_fraction"] for row in rows),
                "mean_initial_aggregate_error": _mean(row["initial_aggregate_morphospace_error"] for row in rows),
                "mean_final_aggregate_error": _mean(row["final_aggregate_morphospace_error"] for row in rows),
                "mean_aggregate_recovery_fraction": _mean(row["aggregate_recovery_fraction"] for row in rows),
                "mean_total_energy_cost": _mean(row["total_energy_cost"] for row in rows),
                "mean_energy_per_site": _mean(row["energy_per_site"] for row in rows),
                "failure_to_improve_rate": _mean(float(row["failure_to_improve"]) for row in rows),
                "low_recovery_failure_rate": _mean(float(row["low_recovery_failure"]) for row in rows),
            }
        )
    by_triplet = {
        (row["task_type"], row["target_kind"], row["policy_id"]): row
        for row in output
        if row["scale_label"] == "small"
    }
    for row in output:
        if row["scale_label"] != "large":
            row["large_to_small_recovery_ratio"] = 1.0
            row["large_to_small_energy_per_site_ratio"] = 1.0
            row["large_to_small_failure_rate_delta"] = 0.0
            continue
        small = by_triplet.get((row["task_type"], row["target_kind"], row["policy_id"]))
        if small is None:
            row["large_to_small_recovery_ratio"] = float("nan")
            row["large_to_small_energy_per_site_ratio"] = float("nan")
            row["large_to_small_failure_rate_delta"] = float("nan")
            continue
        row["large_to_small_recovery_ratio"] = _safe_ratio(
            row["mean_target_recovery_fraction"],
            small["mean_target_recovery_fraction"],
        )
        row["large_to_small_energy_per_site_ratio"] = _safe_ratio(
            row["mean_energy_per_site"],
            small["mean_energy_per_site"],
        )
        row["large_to_small_failure_rate_delta"] = float(row["low_recovery_failure_rate"] - small["low_recovery_failure_rate"])
    return output


def policy_parameter_signature(policy_id: str, event_multiplier: int) -> str:
    return json.dumps(
        {
            "policy_id": policy_id,
            "action_set": "S04 default swap/wait legality and energy semantics, optimized audit-free inner loop",
            "event_multiplier_per_site": int(event_multiplier),
            "policy_code_path": "src/e05/scrambled_embryo.py",
            "large_grid_retuning": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def propose_scaling_action(
    *,
    policy_id: str,
    state: SubstrateState,
    target: TargetMorphology,
    actor_site_id: int,
    rng: random.Random,
    distance_lookup: Mapping[tuple[str, int], float],
) -> MorphogenesisActionRequest:
    """Return the S07 policy proposal, using cached distances for local descent."""

    if policy_id == "s07_random_adjacent_swap_control":
        actor_site_id = int(actor_site_id)
        if state.cell_at(actor_site_id) is None:
            return MorphogenesisActionRequest("wait", actor_site_id)
        neighbors = list(state.neighbors(actor_site_id))
        rng.shuffle(neighbors)
        for neighbor_site_id in neighbors:
            allowed, _reason = state.can_swap(actor_site_id, neighbor_site_id)
            if allowed:
                return MorphogenesisActionRequest("swap", actor_site_id, int(neighbor_site_id))
        return MorphogenesisActionRequest("wait", actor_site_id)
    if policy_id != "s07_local_target_neighbor_descent":
        return policy_by_id(policy_id).propose(state=state, target=target, actor_site_id=actor_site_id, rng=rng)
    actor_site_id = int(actor_site_id)
    if state.cell_at(actor_site_id) is None:
        return MorphogenesisActionRequest("wait", actor_site_id)
    best_target: int | None = None
    best_delta = 0.0
    neighbors = list(state.neighbors(actor_site_id))
    rng.shuffle(neighbors)
    for neighbor_site_id in neighbors:
        allowed, _reason = state.can_swap(actor_site_id, neighbor_site_id)
        if not allowed:
            continue
        delta = cached_adjacent_swap_target_error_delta(
            state,
            actor_site_id,
            int(neighbor_site_id),
            distance_lookup,
        )
        if delta < best_delta - 1e-12:
            best_delta = delta
            best_target = int(neighbor_site_id)
    if best_target is None:
        return MorphogenesisActionRequest("wait", actor_site_id)
    return MorphogenesisActionRequest("swap", actor_site_id, best_target)


def cached_adjacent_swap_target_error_delta(
    state: SubstrateState,
    left_site_id: int,
    right_site_id: int,
    distance_lookup: Mapping[tuple[str, int], float],
) -> float:
    left_site_id = int(left_site_id)
    right_site_id = int(right_site_id)
    left_cell = state.cell_at(left_site_id)
    right_cell = state.cell_at(right_site_id)
    if left_cell is None or right_cell is None:
        return 0.0
    current = (
        distance_lookup[(str(left_cell.cell_id), left_site_id)]
        + distance_lookup[(str(right_cell.cell_id), right_site_id)]
    )
    proposed = (
        distance_lookup[(str(right_cell.cell_id), left_site_id)]
        + distance_lookup[(str(left_cell.cell_id), right_site_id)]
    )
    return float(proposed - current)


def target_identity_distance_lookup(target: TargetMorphology) -> dict[tuple[str, int], float]:
    """Precompute cell-to-target-site identity distances for S09 local descent."""

    constructed = target.constructed_substrate()
    lookup: dict[tuple[str, int], float] = {}
    cells = [constructed.cell_at(site_id) for site_id in target.substrate.site_ids]
    for cell in cells:
        if cell is None:
            continue
        identity = identity_from_substrate_cell(cell, fallback_scalar=False)
        for target_site_id in target.substrate.site_ids:
            lookup[(str(cell.cell_id), int(target_site_id))] = float(
                target.schema.distance(identity, target.identities_by_site[int(target_site_id)])
            )
    return lookup


def execute_swap_wait_fast(
    state: SubstrateState,
    request: MorphogenesisActionRequest,
    *,
    population_current: int,
) -> FastSwapWaitOutcome:
    """Execute S07 swap/wait proposals with S04 legality and energy semantics.

    The full S04 executor records expensive full-state signatures around every
    action.  S09 only uses the S07 controls, which propose `wait` or adjacent
    `swap`, so this fast path preserves the same local legality, population
    conservation, and default energy costs while keeping scaling tests feasible.
    """

    if request.action_type == "wait":
        if request.source_site_id not in state.site_ids:
            return FastSwapWaitOutcome("wait", False, "source_site_missing", False, 0.0, int(population_current))
        if state.cell_at(request.source_site_id) is None:
            return FastSwapWaitOutcome("wait", False, "actor_missing", False, 0.0, int(population_current))
        return FastSwapWaitOutcome("wait", True, "allowed", False, 0.0, int(population_current))
    if request.action_type == "swap":
        if request.target_site_id is None:
            return FastSwapWaitOutcome("swap", False, "target_required", False, 0.0, int(population_current))
        allowed, reason = state.can_swap(request.source_site_id, request.target_site_id)
        if not allowed:
            return FastSwapWaitOutcome("swap", False, reason, False, 0.0, int(population_current))
        state._swap_unchecked(request.source_site_id, request.target_site_id)
        return FastSwapWaitOutcome("swap", True, "allowed", True, 1.0, int(population_current))
    return FastSwapWaitOutcome(request.action_type, False, "unsupported_action", False, 0.0, int(population_current))


def blocked_s08_perturbation_rows() -> tuple[dict[str, Any], ...]:
    """Return S08 perturbations excluded from S09 scaling and why."""

    rows = []
    for perturbation_type in ("missing_patch", "frozen_patch", "duplicated_patch", "foreign_patch"):
        rows.append(
            {
                "research_step_id": "S09",
                "source_research_step_id": "S08",
                "perturbation_type": perturbation_type,
                "included_in_scaling": False,
                "reason": semantic_blocker_for_perturbation(perturbation_type),
            }
        )
    rows.append(
        {
            "research_step_id": "S09",
            "source_research_step_id": "S08",
            "perturbation_type": "rotated_patch",
            "included_in_scaling": True,
            "reason": "S08 classified rotated patches as swap-repairable rearrangements under the S07 controls.",
        }
    )
    return tuple(rows)


def _initial_state_for_task(
    *,
    config: TargetScaleConfig,
    target: TargetMorphology,
    task_type: str,
    seed: int,
) -> tuple[SubstrateState, dict[str, Any]]:
    if task_type == "scrambled_embryo":
        state = scramble_target_state(target, seed)
        return state, {
            "perturbation_type": "full_scramble",
            "mask_id": "",
            "mask_area": 0,
            "mask_area_fraction": 0.0,
            "mask_width": 0,
            "mask_height": 0,
            "baseline_target_error": 0.0,
            "pre_repair_target_error": float(target.target_error(state)),
            "semantic_blocker": "",
            "repairability_class": "permutation_rearrangement",
        }
    perturbation = apply_scaled_rotated_patch(target, config=config, seed=seed)
    return perturbation.state, {
        "perturbation_type": "rotated_patch",
        "mask_id": perturbation.mask.mask_id,
        "mask_area": len(perturbation.mask.site_ids),
        "mask_area_fraction": float(len(perturbation.mask.site_ids) / max(1, len(target.substrate.site_ids))),
        "mask_width": int(perturbation.mask.width),
        "mask_height": int(perturbation.mask.height),
        "baseline_target_error": float(perturbation.baseline_target_error),
        "pre_repair_target_error": float(perturbation.perturbed_target_error),
        "semantic_blocker": "",
        "repairability_class": "swap_repairable_rearrangement",
    }


def apply_scaled_rotated_patch(target: TargetMorphology, *, config: TargetScaleConfig, seed: int) -> ScaledPerturbation:
    """Apply an S08-style 180-degree patch rotation with scaled patch area."""

    mask = select_scaled_patch_mask(target, config=config, seed=seed)
    baseline = target.constructed_substrate()
    cells_by_site = {site_id: baseline.cell_at(site_id) for site_id in baseline.site_ids}
    source_cells_by_site = dict(cells_by_site)
    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    x0, y0 = mask.origin
    x1 = x0 + mask.width - 1
    y1 = y0 + mask.height - 1
    for site_id, (x, y) in zip(mask.site_ids, mask.coordinates, strict=True):
        source_site_id = (y0 + y1 - y) * width + (x0 + x1 - x)
        cells_by_site[site_id] = source_cells_by_site[source_site_id]
    state = target.substrate.copy_empty()
    for site_id in target.substrate.site_ids:
        cell = cells_by_site[site_id]
        if cell is not None:
            state.place_cell(site_id, cell)
    perturbed_metrics = compute_state_metrics(target, state, "initial")
    return ScaledPerturbation(
        mask=mask,
        state=state,
        baseline_target_error=float(target.target_error(baseline)),
        perturbed_target_error=float(perturbed_metrics["target_error"]),
        perturbed_aggregate_morphospace_error=float(perturbed_metrics["aggregate_morphospace_error"]),
        population_baseline=len(target.substrate.site_ids),
        population_perturbed=sum(1 for site_id in target.substrate.site_ids if state.cell_at(site_id) is not None),
    )


def select_scaled_patch_mask(target: TargetMorphology, *, config: TargetScaleConfig, seed: int) -> PatchMask:
    """Select a deterministic rotated-patch mask with area scaled from S08 defaults."""

    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    height = int(target.substrate.dimensions.get("height", 1))
    base_width = min(3, config.scaled_from_width)
    base_height = min(3, config.scaled_from_height)
    if config.scaled_from_height <= 4:
        base_height = min(2, config.scaled_from_height)
    patch_width = min(width, max(1, int(round(base_width * config.scale_factor))))
    patch_height = min(height, max(1, int(round(base_height * config.scale_factor))))
    rng = random.Random(int(seed) + stable_int_hash(f"S09:{config.config_id}:scaled_rotated_patch"))
    center_x = max(0, (width - patch_width) // 2)
    center_y = max(0, (height - patch_height) // 2)
    jitter_span = max(1, int(round(config.scale_factor)))
    jitter_x = rng.choice((-jitter_span, 0, jitter_span)) if width > patch_width else 0
    jitter_y = rng.choice((-jitter_span, 0, jitter_span)) if height > patch_height else 0
    origin_x = min(max(0, center_x + jitter_x), width - patch_width)
    origin_y = min(max(0, center_y + jitter_y), height - patch_height)
    coordinates = tuple(
        (x, y)
        for y in range(origin_y, origin_y + patch_height)
        for x in range(origin_x, origin_x + patch_width)
    )
    site_ids = tuple(y * width + x for x, y in coordinates)
    mask_id = f"{target.target_id}:scaled_rotated_patch:seed_{int(seed)}:{origin_x}_{origin_y}_{patch_width}x{patch_height}"
    return PatchMask(
        mask_id=mask_id,
        site_ids=site_ids,
        coordinates=coordinates,
        origin=(origin_x, origin_y),
        width=patch_width,
        height=patch_height,
        seed=int(seed),
    )


def _tag_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TargetScaleConfig,
    task_type: str,
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
                "research_step_id": "S09",
                "task_type": task_type,
                "config_id": config.config_id,
                "source_target_id": config.source_target_id,
                "target_kind": config.target_kind,
                "scale_label": config.scale_label,
                "width": int(config.width),
                "height": int(config.height),
                "site_count": config.site_count,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "simulation_seed": int(seed),
                "benchmark_state_label": state_label,
            }
        )
        tagged.append(payload)
    return tagged


def _target_constructor(constructor_name: str) -> Callable[..., TargetMorphology]:
    constructors: dict[str, Callable[..., TargetMorphology]] = {
        "gradient_target": gradient_target,
        "stripes_target": stripes_target,
        "boundary_target": boundary_target,
        "ring_target": ring_target,
        "symmetry_target": symmetry_target,
        "organ_like_target": organ_like_target,
    }
    if constructor_name not in constructors:
        raise KeyError(f"unsupported scaling target constructor: {constructor_name}")
    return constructors[constructor_name]


def _scaled_dimension(value: int, target_kind: str, *, axis: str) -> int:
    scaled = int(value) * 2
    if target_kind == "symmetry" and axis == "width" and scaled % 2 == 0:
        scaled += 1
    return scaled


def _validate_task_types(task_types: Sequence[str]) -> None:
    unsupported = sorted(set(task_types) - set(SCALING_TASK_TYPES))
    if unsupported:
        raise ValueError(f"unsupported S09 task types: {unsupported}")


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return float("inf") if float(numerator) > 0.0 else 1.0
    return float(float(numerator) / denominator)
