"""Local-versus-global control baselines for E05 S13."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.e05.cell_identity import CellIdentity, identity_from_substrate_cell
from src.e05.morphospace_metrics import aggregate_morphospace_error, evaluate_morphology_metrics
from src.e05.regeneration import (
    DEFAULT_REGENERATION_SEEDS,
    PERTURBATION_TYPES,
    STUCK_STATUS,
    apply_regeneration_perturbation,
    identity_multiset_matches_target,
    repairability_class,
    semantic_blocker_for_perturbation,
)
from src.e05.scaling import (
    DEFAULT_SCALING_SEEDS,
    DEFAULT_EVENT_MULTIPLIER as S09_EVENT_MULTIPLIER,
    DEFAULT_RECORDS_PER_RUN as S09_RECORDS_PER_RUN,
    SCALING_TASK_TYPES,
    TargetScaleConfig,
    _initial_state_for_task,
    default_scaling_target_configs,
)
from src.e05.scrambled_embryo import (
    DEFAULT_EVENT_MULTIPLIER as S07_EVENT_MULTIPLIER,
    DEFAULT_RECORDS_PER_RUN as S07_RECORDS_PER_RUN,
    DEFAULT_SEEDS,
    RUNNABLE_POLICY_IDS,
    clone_substrate_state,
    compute_state_metrics,
    recovery_fraction,
    scramble_displacement_fraction,
    scramble_target_state,
    two_dimensional_targets,
)
from src.e05.substrates import SubstrateCell, SubstrateState
from src.e05.symmetry_breaking import (
    DEFAULT_EVENT_MULTIPLIER as S10_EVENT_MULTIPLIER,
    DEFAULT_RECORDS_PER_RUN as S10_RECORDS_PER_RUN,
    DEFAULT_SYMMETRY_SEEDS,
    axis_choice_metrics,
    construct_axis_symmetric_initial_state,
    default_initial_conditions,
    default_symmetry_target,
)
from src.e05.targets import TargetMorphology, default_target_gallery


STEP_ID = "S13"
LOCAL_INFORMATION_ACCESS_TIER = "local_only"
LOW_RECOVERY_FAILURE_THRESHOLD = 0.05


@dataclass(frozen=True)
class BaselineSpec:
    """One S13 nonlocal baseline with explicit information and action access."""

    policy_id: str
    policy_family: str
    control_class: str
    declared_information_access: str
    baseline_description: str
    global_state_access: bool
    global_target_access: bool
    organizer_cue_added: bool
    nonlocal_relocation_allowed: bool
    unfreeze_allowed: bool
    birth_death_allowed: bool
    identity_conversion_allowed: bool
    energy_cost_model: str

    @property
    def local_policy_information_access_changed(self) -> bool:
        return False

    def compact_dict(self) -> dict[str, Any]:
        return {
            "research_step_id": STEP_ID,
            "policy_id": self.policy_id,
            "policy_family": self.policy_family,
            "control_class": self.control_class,
            "information_access_tier": self.control_class,
            "declared_information_access": self.declared_information_access,
            "baseline_description": self.baseline_description,
            "global_state_access": bool(self.global_state_access),
            "global_target_access": bool(self.global_target_access),
            "organizer_cue_added": bool(self.organizer_cue_added),
            "nonlocal_relocation_allowed": bool(self.nonlocal_relocation_allowed),
            "unfreeze_allowed": bool(self.unfreeze_allowed),
            "birth_death_allowed": bool(self.birth_death_allowed),
            "identity_conversion_allowed": bool(self.identity_conversion_allowed),
            "local_policy_information_access_changed": False,
            "energy_cost_model": self.energy_cost_model,
        }


@dataclass(frozen=True)
class AssignmentPlan:
    """Exact identity-preserving assignment plan for existing cells."""

    feasible: bool
    final_state: SubstrateState | None
    blocked_reason: str
    nonlocal_relocation_count: int
    global_assignment_distance: float
    unfreeze_count: int


def baseline_specs() -> tuple[BaselineSpec, ...]:
    """Return the S13 nonlocal controls with claim-boundary labels."""

    return (
        BaselineSpec(
            policy_id="s13_top_down_global_assignment",
            policy_family="s13_top_down_existing_identity_assignment",
            control_class="top_down",
            declared_information_access=(
                "full tissue state, full target identity map, and nonlocal assignment planner; no birth/death, "
                "identity conversion, or unfreeze authority"
            ),
            baseline_description=(
                "Upper-bound top-down planner that globally relocates existing active cells to exact target sites "
                "when the identity multiset already matches the target."
            ),
            global_state_access=True,
            global_target_access=True,
            organizer_cue_added=False,
            nonlocal_relocation_allowed=True,
            unfreeze_allowed=False,
            birth_death_allowed=False,
            identity_conversion_allowed=False,
            energy_cost_model="sum of Manhattan distances for nonlocal identity-preserving relocations",
        ),
        BaselineSpec(
            policy_id="s13_organizer_beacon_assignment",
            policy_family="s13_organizer_field_existing_identity_assignment",
            control_class="organizer",
            declared_information_access=(
                "organizer-provided global target field plus full tissue state and nonlocal assignment planner; "
                "may unfreeze stuck cells but cannot synthesize or convert identities"
            ),
            baseline_description=(
                "Organizer-field baseline that treats target-site identities as a broadcast morphogen-like cue and "
                "permits unfreezing before exact identity-preserving global assignment."
            ),
            global_state_access=True,
            global_target_access=True,
            organizer_cue_added=True,
            nonlocal_relocation_allowed=True,
            unfreeze_allowed=True,
            birth_death_allowed=False,
            identity_conversion_allowed=False,
            energy_cost_model="Manhattan relocation distance plus one unit per unfreeze event",
        ),
        BaselineSpec(
            policy_id="s13_global_rebuild_controller",
            policy_family="s13_global_target_state_rebuild",
            control_class="global_controller",
            declared_information_access=(
                "full tissue state, full target identity map, and target-state overwrite authority including "
                "birth/death, identity conversion, and unfreeze semantics"
            ),
            baseline_description=(
                "Centralized rebuild upper bound that constructs the exact target state even when local swap/wait "
                "semantics cannot repair population deficits or identity-multiset conflicts."
            ),
            global_state_access=True,
            global_target_access=True,
            organizer_cue_added=True,
            nonlocal_relocation_allowed=True,
            unfreeze_allowed=True,
            birth_death_allowed=True,
            identity_conversion_allowed=True,
            energy_cost_model=(
                "site overwrite proxy: one unit per site plus identity edit, population delta, and unfreeze costs"
            ),
        ),
    )


def baseline_spec_rows() -> list[dict[str, Any]]:
    return [spec.compact_dict() for spec in baseline_specs()]


def build_s13_result_tables(artifacts_dir: Path) -> dict[str, pd.DataFrame]:
    """Build local/global comparison, gap, scaling, robustness, and validation tables."""

    artifacts_dir = Path(artifacts_dir)
    local_df = load_local_control_rows(artifacts_dir)
    global_df = pd.DataFrame(build_global_control_rows())
    route_context_df = load_s12_route_context(artifacts_dir)
    control_df = pd.concat([local_df, global_df], ignore_index=True, sort=False)
    control_df = add_failure_and_scaling_fields(control_df)
    control_df = attach_route_context(control_df, route_context_df)
    gap_df = compute_gap_summary(control_df)
    robustness_df = compute_robustness_summary(control_df)
    scaling_df = compute_scaling_gap_summary(control_df)
    validation_df = validate_s13_tables(
        control_df=control_df,
        gap_df=gap_df,
        robustness_df=robustness_df,
        scaling_df=scaling_df,
        route_context_df=route_context_df,
    )
    return {
        "control": control_df,
        "baseline_specs": pd.DataFrame(baseline_spec_rows()),
        "gap_summary": gap_df,
        "robustness_summary": robustness_df,
        "scaling_summary": scaling_df,
        "route_context": route_context_df,
        "validation": validation_df,
    }


def load_local_control_rows(artifacts_dir: Path) -> pd.DataFrame:
    """Normalize S07-S10 local-policy result rows into the S13 schema."""

    artifacts_dir = Path(artifacts_dir)
    frames = [
        _local_s07_rows(pd.read_parquet(artifacts_dir / "results" / "e05_scrambled_embryo_results.parquet")),
        _local_s08_rows(pd.read_parquet(artifacts_dir / "results" / "e05_regeneration_results.parquet")),
        _local_s09_rows(pd.read_parquet(artifacts_dir / "results" / "e05_scaling_tests.parquet")),
        _local_s10_rows(pd.read_parquet(artifacts_dir / "results" / "e05_symmetry_breaking.parquet")),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def build_global_control_rows() -> list[dict[str, Any]]:
    """Evaluate S13 global/top-down baselines on S07-S10 task starts."""

    rows: list[dict[str, Any]] = []
    for spec in baseline_specs():
        rows.extend(_global_s07_rows(spec))
        rows.extend(_global_s08_rows(spec))
        rows.extend(_global_s09_rows(spec))
        rows.extend(_global_s10_rows(spec))
    return rows


def evaluate_baseline_on_state(
    *,
    spec: BaselineSpec,
    target: TargetMorphology,
    initial_state: SubstrateState,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one standardized S13 global-baseline run row."""

    initial_metrics = compute_state_metrics(target, initial_state, "initial")
    initial_target_error = float(initial_metrics["target_error"])
    initial_aggregate_error = float(initial_metrics["aggregate_morphospace_error"])
    initial_population = _occupied_count(initial_state)
    initial_stuck = _stuck_count(initial_state)
    final_state: SubstrateState
    blocked = False
    blocked_reason = ""
    nonlocal_relocations = 0
    assignment_distance = 0.0
    unfreeze_count = 0
    identity_conversion_count = 0
    birth_count = 0
    death_count = 0
    total_energy = 0.0

    if spec.identity_conversion_allowed or spec.birth_death_allowed:
        final_state = target.constructed_substrate()
        birth_count = max(0, len(target.substrate.site_ids) - initial_population)
        death_count = max(0, initial_population - len(target.substrate.site_ids))
        identity_conversion_count = _identity_edit_count(target, initial_state)
        unfreeze_count = initial_stuck if spec.unfreeze_allowed else 0
        total_energy = float(
            len(target.substrate.site_ids)
            + identity_conversion_count
            + birth_count
            + death_count
            + unfreeze_count
        )
    else:
        plan = exact_identity_assignment_plan(target, initial_state, allow_unfreeze=spec.unfreeze_allowed)
        if plan.feasible and plan.final_state is not None:
            final_state = plan.final_state
            nonlocal_relocations = plan.nonlocal_relocation_count
            assignment_distance = plan.global_assignment_distance
            unfreeze_count = plan.unfreeze_count
            total_energy = float(assignment_distance + unfreeze_count)
        else:
            final_state = clone_substrate_state(initial_state)
            blocked = True
            blocked_reason = plan.blocked_reason or "assignment_infeasible"

    final_metrics = compute_state_metrics(target, final_state, "final")
    final_target_error = float(final_metrics["target_error"])
    final_aggregate_error = float(final_metrics["aggregate_morphospace_error"])
    site_count = int(len(target.substrate.site_ids))
    row = {
        "research_step_id": STEP_ID,
        "source_research_step_id": metadata.get("source_research_step_id", ""),
        "benchmark_task": metadata.get("benchmark_task", ""),
        "task_id": metadata.get("task_id", ""),
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "target_hash": target.target_hash,
        "substrate_kind": target.substrate.kind,
        "site_count": site_count,
        "policy_id": spec.policy_id,
        "policy_family": spec.policy_family,
        "control_class": spec.control_class,
        "information_access_tier": spec.control_class,
        "declared_information_access": spec.declared_information_access,
        "baseline_description": spec.baseline_description,
        "global_state_access": bool(spec.global_state_access),
        "global_target_access": bool(spec.global_target_access),
        "organizer_cue_added": bool(spec.organizer_cue_added),
        "nonlocal_relocation_allowed": bool(spec.nonlocal_relocation_allowed),
        "unfreeze_allowed": bool(spec.unfreeze_allowed),
        "birth_death_allowed": bool(spec.birth_death_allowed),
        "identity_conversion_allowed": bool(spec.identity_conversion_allowed),
        "local_policy_information_access_changed": False,
        "simulation_seed": int(metadata.get("simulation_seed", -1)),
        "event_cap": int(metadata.get("event_cap", 0)),
        "event_multiplier": int(metadata.get("event_multiplier", 0)),
        "records_per_run": int(metadata.get("records_per_run", 0)),
        "retuning_allowed": bool(metadata.get("retuning_allowed", False)),
        "large_grid_retuned": bool(metadata.get("large_grid_retuned", False)),
        "tuned_on_scale_label": metadata.get("tuned_on_scale_label", ""),
        "initial_target_error": initial_target_error,
        "final_target_error": final_target_error,
        "delta_target_error": final_target_error - initial_target_error,
        "target_recovery_fraction": recovery_fraction(initial_target_error, final_target_error),
        "initial_aggregate_morphospace_error": initial_aggregate_error,
        "final_aggregate_morphospace_error": final_aggregate_error,
        "delta_aggregate_morphospace_error": final_aggregate_error - initial_aggregate_error,
        "aggregate_recovery_fraction": recovery_fraction(initial_aggregate_error, final_aggregate_error),
        "initial_displacement_fraction": scramble_displacement_fraction(target, initial_state),
        "final_displacement_fraction": scramble_displacement_fraction(target, final_state),
        "accepted_swaps": 0,
        "attempted_swaps": 0,
        "rejected_actions": 0,
        "wait_actions": 0,
        "nonlocal_relocations": int(nonlocal_relocations),
        "global_assignment_distance": float(assignment_distance),
        "identity_conversion_count": int(identity_conversion_count),
        "birth_count": int(birth_count),
        "death_count": int(death_count),
        "unfreeze_count": int(unfreeze_count),
        "total_energy_cost": float(total_energy),
        "energy_per_site": float(total_energy / max(1, site_count)),
        "energy_cost_model": spec.energy_cost_model,
        "population_initial": int(initial_population),
        "population_final": int(_occupied_count(final_state)),
        "population_min": int(min(initial_population, _occupied_count(final_state))),
        "population_max": int(max(initial_population, _occupied_count(final_state))),
        "population_delta_total": int(_occupied_count(final_state) - initial_population),
        "identity_multiset_matches_target_initial": bool(identity_multiset_matches_target(target, initial_state)),
        "identity_multiset_matches_target_final": bool(identity_multiset_matches_target(target, final_state)),
        "stuck_site_count_initial": int(initial_stuck),
        "stuck_site_count_final": int(_stuck_count(final_state)),
        "empty_site_count_initial": int(site_count - initial_population),
        "empty_site_count_final": int(site_count - _occupied_count(final_state)),
        "baseline_blocked": bool(blocked),
        "blocked_reason": blocked_reason,
        "semantic_blocker": metadata.get("semantic_blocker", ""),
        "repairability_class": metadata.get("repairability_class", ""),
        "scale_label": metadata.get("scale_label", ""),
        "scale_factor": float(metadata.get("scale_factor", np.nan)),
        "config_id": metadata.get("config_id", ""),
        "source_target_id": metadata.get("source_target_id", ""),
        "task_type": metadata.get("task_type", ""),
        "perturbation_type": metadata.get("perturbation_type", ""),
        "mask_id": metadata.get("mask_id", ""),
        "mask_area": int(metadata.get("mask_area", 0)),
        "mask_area_fraction": float(metadata.get("mask_area_fraction", 0.0)),
        "condition_id": metadata.get("condition_id", ""),
        "construction_axis": metadata.get("construction_axis", ""),
        "symmetry_class": metadata.get("symmetry_class", ""),
        "near_symmetry_break_swaps": int(metadata.get("near_symmetry_break_swaps", 0)),
        "source_row_count": 1,
    }
    if metadata.get("benchmark_task") == "symmetry_breaking":
        row.update(_axis_summary_fields(target, initial_state, final_state, metadata))
    return row


def exact_identity_assignment_plan(
    target: TargetMorphology,
    state: SubstrateState,
    *,
    allow_unfreeze: bool,
) -> AssignmentPlan:
    """Return an exact identity-preserving assignment if current cells can fill the target."""

    expected_counter = Counter(_identity_signature(identity) for identity in target.identities_by_site.values())
    observed_counter = Counter()
    sources_by_key: dict[str, list[tuple[int, SubstrateCell]]] = {}
    for site_id in state.site_ids:
        cell = state.cell_at(site_id)
        if cell is None:
            continue
        try:
            identity = identity_from_substrate_cell(cell, fallback_scalar=False)
        except ValueError:
            return AssignmentPlan(False, None, "cell_missing_e05_identity", 0, 0.0, 0)
        key = _identity_signature(identity)
        observed_counter[key] += 1
        sources_by_key.setdefault(key, []).append((int(site_id), cell))
    if observed_counter != expected_counter:
        return AssignmentPlan(False, None, "identity_multiset_or_population_mismatch", 0, 0.0, 0)

    targets_by_key: dict[str, list[int]] = {}
    for site_id, identity in target.identities_by_site.items():
        targets_by_key.setdefault(_identity_signature(identity), []).append(int(site_id))

    assignments: list[tuple[int, int, SubstrateCell]] = []
    unfreeze_count = 0
    for key, targets in targets_by_key.items():
        sources = list(sources_by_key.get(key, []))
        remaining_targets = list(targets)
        remaining_sources: list[tuple[int, SubstrateCell]] = []
        for source_site_id, cell in sources:
            if _is_stuck(cell) and not allow_unfreeze:
                if source_site_id in remaining_targets:
                    assignments.append((source_site_id, source_site_id, cell))
                    remaining_targets.remove(source_site_id)
                else:
                    return AssignmentPlan(False, None, "unfreeze_required_for_stuck_misplaced_cell", 0, 0.0, 0)
            else:
                remaining_sources.append((source_site_id, cell))
        if len(remaining_sources) != len(remaining_targets):
            return AssignmentPlan(False, None, "assignment_cardinality_mismatch", 0, 0.0, 0)
        if remaining_sources:
            cost = np.array(
                [
                    [_coordinate_distance(target.substrate, source_site_id, target_site_id) for target_site_id in remaining_targets]
                    for source_site_id, _cell in remaining_sources
                ],
                dtype=float,
            )
            source_indices, target_indices = linear_sum_assignment(cost)
            for source_idx, target_idx in zip(source_indices, target_indices, strict=True):
                source_site_id, cell = remaining_sources[int(source_idx)]
                target_site_id = remaining_targets[int(target_idx)]
                if _is_stuck(cell) and allow_unfreeze and source_site_id != target_site_id:
                    unfreeze_count += 1
                assignments.append((source_site_id, target_site_id, cell))

    final_state = target.substrate.copy_empty()
    relocation_count = 0
    distance = 0.0
    for source_site_id, target_site_id, cell in assignments:
        placed = _with_active_status(cell) if allow_unfreeze and _is_stuck(cell) else cell
        final_state.place_cell(target_site_id, placed)
        if source_site_id != target_site_id:
            relocation_count += 1
            distance += _coordinate_distance(target.substrate, source_site_id, target_site_id)
    return AssignmentPlan(True, final_state, "", int(relocation_count), float(distance), int(unfreeze_count))


def compute_gap_summary(control_df: pd.DataFrame) -> pd.DataFrame:
    """Quantify recovery/error/energy gaps against the local target-aware control."""

    group_cols = ["benchmark_task", "target_kind", "policy_id", "control_class", "scale_label"]
    rows = []
    grouped = (
        control_df.groupby(group_cols, dropna=False)
        .agg(
            runs=("task_id", "count"),
            mean_initial_target_error=("initial_target_error", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            mean_final_aggregate_error=("final_aggregate_morphospace_error", "mean"),
            mean_aggregate_recovery_fraction=("aggregate_recovery_fraction", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
            blocked_rate=("baseline_blocked", "mean"),
        )
        .reset_index()
    )
    local_ref = grouped[grouped["policy_id"] == "s07_local_target_neighbor_descent"].copy()
    ref_map = {
        (row["benchmark_task"], row["target_kind"], row["scale_label"]): row
        for row in local_ref.to_dict(orient="records")
    }
    for row in grouped.to_dict(orient="records"):
        ref = ref_map.get((row["benchmark_task"], row["target_kind"], row["scale_label"]))
        payload = {"research_step_id": STEP_ID, **row}
        if ref is None:
            payload.update(
                {
                    "target_recovery_gap_vs_local_target": np.nan,
                    "final_target_error_gap_vs_local_target": np.nan,
                    "energy_per_site_gap_vs_local_target": np.nan,
                    "low_recovery_failure_rate_gap_vs_local_target": np.nan,
                }
            )
        else:
            payload.update(
                {
                    "target_recovery_gap_vs_local_target": float(
                        row["mean_target_recovery_fraction"] - ref["mean_target_recovery_fraction"]
                    ),
                    "final_target_error_gap_vs_local_target": float(
                        row["mean_final_target_error"] - ref["mean_final_target_error"]
                    ),
                    "energy_per_site_gap_vs_local_target": float(row["mean_energy_per_site"] - ref["mean_energy_per_site"]),
                    "low_recovery_failure_rate_gap_vs_local_target": float(
                        row["low_recovery_failure_rate"] - ref["low_recovery_failure_rate"]
                    ),
                }
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def compute_robustness_summary(control_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize failure rates by task and information-access tier."""

    grouped = (
        control_df.groupby(["benchmark_task", "control_class", "policy_id"], dropna=False)
        .agg(
            runs=("task_id", "count"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            median_target_recovery_fraction=("target_recovery_fraction", "median"),
            mean_final_target_error=("final_target_error", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
            failure_to_improve_rate=("failure_to_improve", "mean"),
            blocked_rate=("baseline_blocked", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "research_step_id", STEP_ID)
    return grouped


def compute_scaling_gap_summary(control_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize small-to-large S09 transfer gaps for local and global controls."""

    s09 = control_df[control_df["source_research_step_id"].eq("S09")].copy()
    if s09.empty:
        return pd.DataFrame()
    grouped = (
        s09.groupby(["task_type", "target_kind", "policy_id", "control_class", "scale_label"], dropna=False)
        .agg(
            runs=("task_id", "count"),
            site_count=("site_count", "mean"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
            blocked_rate=("baseline_blocked", "mean"),
        )
        .reset_index()
    )
    output = []
    small_map = {
        (row["task_type"], row["target_kind"], row["policy_id"]): row
        for row in grouped[grouped["scale_label"].eq("small")].to_dict(orient="records")
    }
    for row in grouped.to_dict(orient="records"):
        small = small_map.get((row["task_type"], row["target_kind"], row["policy_id"]))
        payload = {"research_step_id": STEP_ID, **row}
        if row["scale_label"] == "large" and small is not None:
            payload["large_to_small_recovery_ratio"] = _safe_ratio(
                row["mean_target_recovery_fraction"],
                small["mean_target_recovery_fraction"],
            )
            payload["large_to_small_energy_per_site_ratio"] = _safe_ratio(
                row["mean_energy_per_site"],
                small["mean_energy_per_site"],
            )
            payload["large_to_small_failure_rate_delta"] = float(
                row["low_recovery_failure_rate"] - small["low_recovery_failure_rate"]
            )
        else:
            payload["large_to_small_recovery_ratio"] = 1.0
            payload["large_to_small_energy_per_site_ratio"] = 1.0
            payload["large_to_small_failure_rate_delta"] = 0.0
        output.append(payload)
    return pd.DataFrame(output)


def load_s12_route_context(artifacts_dir: Path) -> pd.DataFrame:
    """Load S12 route metrics as route context for local policies."""

    path = Path(artifacts_dir) / "results" / "e05_morphospace_route_metrics.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    rows = []
    for row in df.to_dict(orient="records"):
        source = str(row["source_research_step_id"])
        task_id = str(row["task_id"])
        benchmark_task = _benchmark_task_from_s12(source, str(row.get("source_task_type", "")))
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": source,
                "benchmark_task": benchmark_task,
                "task_id": _s13_task_id_from_s12(source, task_id, row),
                "target_id": row["target_id"],
                "policy_id": row["policy_id"],
                "simulation_seed": int(row["simulation_seed"]),
                "s12_path_curvature": float(row["path_curvature"]),
                "s12_temporary_away_fraction": float(row["temporary_away_fraction"]),
                "s12_route_diversity_to_group_mean": float(row["route_diversity_to_group_mean"]),
                "s12_group_mean_pairwise_route_distance": float(row["group_mean_pairwise_route_distance"]),
            }
        )
    return pd.DataFrame(rows)


def attach_route_context(control_df: pd.DataFrame, route_context_df: pd.DataFrame) -> pd.DataFrame:
    if route_context_df.empty:
        output = control_df.copy()
        for column in (
            "s12_path_curvature",
            "s12_temporary_away_fraction",
            "s12_route_diversity_to_group_mean",
            "s12_group_mean_pairwise_route_distance",
        ):
            output[column] = np.nan
        return output
    route_context_df = route_context_df.drop(columns=["research_step_id"], errors="ignore")
    merged = control_df.merge(
        route_context_df,
        on=["source_research_step_id", "benchmark_task", "task_id", "target_id", "policy_id", "simulation_seed"],
        how="left",
    )
    return merged


def validate_s13_tables(
    *,
    control_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    route_context_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return S13 validation rows."""

    local = control_df[control_df["control_class"].eq("local")]
    global_rows = control_df[~control_df["control_class"].eq("local")]
    rows = [
        _validation_row(
            "local_policy_information_access_unchanged",
            bool(
                not local.empty
                and set(local["policy_id"].dropna().unique()).issubset(set(RUNNABLE_POLICY_IDS))
                and not local["global_state_access"].any()
                and not local["global_target_access"].any()
                and not local["local_policy_information_access_changed"].any()
            ),
            {"local_policy_ids": list(RUNNABLE_POLICY_IDS), "global_access": False},
            {
                "local_rows": int(len(local)),
                "local_policy_ids": sorted(str(value) for value in local["policy_id"].dropna().unique()),
                "any_local_global_state_access": bool(local["global_state_access"].any()) if not local.empty else None,
            },
            "Local S07-S10 rows were imported without adding global or organizer access.",
        ),
        _validation_row(
            "global_baselines_labeled",
            bool(
                {"top_down", "organizer", "global_controller"}.issubset(set(global_rows["control_class"]))
                and global_rows["global_state_access"].all()
                and global_rows["global_target_access"].all()
            ),
            {"control_classes": ["top_down", "organizer", "global_controller"], "global_access": True},
            {
                "global_rows": int(len(global_rows)),
                "control_classes": sorted(str(value) for value in global_rows["control_class"].dropna().unique()),
            },
            "S13 global controls are explicit nonlocal baselines, not altered local policies.",
        ),
        _validation_row(
            "global_rebuild_solves_all_task_starts",
            bool(
                not control_df[control_df["policy_id"].eq("s13_global_rebuild_controller")].empty
                and (
                    control_df.loc[
                        control_df["policy_id"].eq("s13_global_rebuild_controller"),
                        "final_target_error",
                    ]
                    <= 1e-12
                ).all()
            ),
            {"max_final_target_error": 0.0},
            {
                "max_final_target_error": float(
                    control_df.loc[
                        control_df["policy_id"].eq("s13_global_rebuild_controller"),
                        "final_target_error",
                    ].max()
                )
            },
            "The centralized rebuild upper bound constructs exact target states, including blocked S08 perturbations.",
        ),
        _validation_row(
            "assignment_baselines_record_blockers",
            bool(
                control_df.loc[
                    control_df["policy_id"].isin(
                        ["s13_top_down_global_assignment", "s13_organizer_beacon_assignment"]
                    )
                    & control_df["benchmark_task"].eq("regeneration")
                    & control_df["perturbation_type"].isin(["missing_patch", "duplicated_patch", "foreign_patch"]),
                    "baseline_blocked",
                ].all()
            ),
            {"identity_multiset_blockers_recorded": True},
            {
                "blocked_rows": int(
                    control_df[
                        control_df["baseline_blocked"]
                        & control_df["policy_id"].isin(
                            ["s13_top_down_global_assignment", "s13_organizer_beacon_assignment"]
                        )
                    ].shape[0]
                )
            },
            "Identity-preserving global assignment baselines do not silently solve missing, duplicated, or foreign identity-multiset conflicts.",
        ),
        _validation_row(
            "recovery_error_energy_gaps_quantified",
            bool(
                not gap_df.empty
                and {"target_recovery_gap_vs_local_target", "final_target_error_gap_vs_local_target", "energy_per_site_gap_vs_local_target"}.issubset(
                    gap_df.columns
                )
                and gap_df["target_recovery_gap_vs_local_target"].notna().any()
            ),
            {"gap_columns_present": True},
            {"gap_rows": int(len(gap_df))},
            "Recovery, final-error, and energy-per-site gaps are reported against the local target-aware control.",
        ),
        _validation_row(
            "robustness_gaps_quantified",
            bool(not robustness_df.empty and robustness_df["low_recovery_failure_rate"].between(0.0, 1.0).all()),
            {"failure_rates_bounded": True},
            {"robustness_rows": int(len(robustness_df))},
            "Robustness is summarized by low-recovery failure and failure-to-improve rates.",
        ),
        _validation_row(
            "scaling_gaps_quantified",
            bool(
                not scaling_df.empty
                and {"small", "large"}.issubset(set(scaling_df["scale_label"]))
                and scaling_df["large_to_small_recovery_ratio"].notna().any()
            ),
            {"scale_labels": ["small", "large"]},
            {
                "scaling_rows": int(len(scaling_df)),
                "scale_labels": sorted(str(value) for value in scaling_df["scale_label"].dropna().unique()),
            },
            "S09 small/large scaling gaps are reported for local and global controls.",
        ),
        _validation_row(
            "s12_route_context_loaded",
            bool(not route_context_df.empty and route_context_df["s12_path_curvature"].notna().any()),
            {"s12_route_context_rows": ">0"},
            {"route_context_rows": int(len(route_context_df))},
            "S12 route curvature and temporary-away metrics are loaded as local-policy context.",
        ),
        _validation_row(
            "source_task_coverage_complete",
            bool({"S07", "S08", "S09", "S10"}.issubset(set(control_df["source_research_step_id"]))),
            {"source_steps": ["S07", "S08", "S09", "S10"]},
            {"source_steps": sorted(str(value) for value in control_df["source_research_step_id"].dropna().unique())},
            "S13 comparison covers scrambled embryo, regeneration, scaling, and symmetry-breaking tasks.",
        ),
    ]
    return pd.DataFrame(rows)


def add_failure_and_scaling_fields(control_df: pd.DataFrame) -> pd.DataFrame:
    output = control_df.copy()
    for column in ("baseline_blocked", "global_state_access", "global_target_access", "local_policy_information_access_changed"):
        if column not in output.columns:
            output[column] = False
        output[column] = output[column].fillna(False).astype(bool)
    if "energy_per_site" not in output.columns:
        output["energy_per_site"] = output["total_energy_cost"] / output["site_count"].clip(lower=1)
    output["failure_to_improve"] = output["final_target_error"] >= output["initial_target_error"] - 1e-12
    output["low_recovery_failure"] = output["target_recovery_fraction"] < LOW_RECOVERY_FAILURE_THRESHOLD
    output.loc[output["baseline_blocked"], "low_recovery_failure"] = True
    output["scale_label"] = output.get("scale_label", "").fillna("").astype(str)
    output["control_group"] = np.where(output["control_class"].eq("local"), "local_controls", "nonlocal_baselines")
    return output


def summarize_for_report(control_df: pd.DataFrame, gap_df: pd.DataFrame) -> dict[str, Any]:
    """Return compact S13 report values."""

    policy_summary = (
        control_df.groupby(["control_class", "policy_id"], dropna=False)
        .agg(
            runs=("task_id", "count"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
            blocked_rate=("baseline_blocked", "mean"),
        )
        .reset_index()
    )
    local_target = policy_summary[policy_summary["policy_id"].eq("s07_local_target_neighbor_descent")]
    global_rebuild = policy_summary[policy_summary["policy_id"].eq("s13_global_rebuild_controller")]
    return {
        "control_rows": int(len(control_df)),
        "gap_rows": int(len(gap_df)),
        "policy_summary": policy_summary,
        "local_target_mean_recovery": float(local_target["mean_target_recovery_fraction"].iloc[0])
        if not local_target.empty
        else float("nan"),
        "global_rebuild_mean_recovery": float(global_rebuild["mean_target_recovery_fraction"].iloc[0])
        if not global_rebuild.empty
        else float("nan"),
    }


def _global_s07_rows(spec: BaselineSpec) -> list[dict[str, Any]]:
    rows = []
    for target in two_dimensional_targets(default_target_gallery()):
        for seed in DEFAULT_SEEDS:
            initial_state = scramble_target_state(target, int(seed))
            rows.append(
                evaluate_baseline_on_state(
                    spec=spec,
                    target=target,
                    initial_state=initial_state,
                    metadata={
                        "source_research_step_id": "S07",
                        "benchmark_task": "scrambled_embryo",
                        "task_id": f"S07:{target.target_id}",
                        "simulation_seed": int(seed),
                        "event_cap": S07_EVENT_MULTIPLIER * len(target.substrate.site_ids),
                        "event_multiplier": S07_EVENT_MULTIPLIER,
                        "records_per_run": S07_RECORDS_PER_RUN,
                        "retuning_allowed": False,
                        "perturbation_type": "full_scramble",
                        "repairability_class": "permutation_rearrangement",
                    },
                )
            )
    return rows


def _global_s08_rows(spec: BaselineSpec) -> list[dict[str, Any]]:
    rows = []
    targets = two_dimensional_targets(default_target_gallery())
    for target in targets:
        for perturbation_type in PERTURBATION_TYPES:
            for seed in DEFAULT_REGENERATION_SEEDS:
                perturbed = apply_regeneration_perturbation(
                    target,
                    perturbation_type,
                    int(seed),
                    donor_targets=targets,
                )
                rows.append(
                    evaluate_baseline_on_state(
                        spec=spec,
                        target=target,
                        initial_state=perturbed.state,
                        metadata={
                            "source_research_step_id": "S08",
                            "benchmark_task": "regeneration",
                            "task_id": f"S08:{target.target_id}:{perturbation_type}",
                            "simulation_seed": int(seed),
                            "event_cap": 50 * len(target.substrate.site_ids),
                            "event_multiplier": 50,
                            "records_per_run": 25,
                            "retuning_allowed": False,
                            "perturbation_type": perturbation_type,
                            "mask_id": perturbed.mask.mask_id,
                            "mask_area": len(perturbed.mask.site_ids),
                            "mask_area_fraction": len(perturbed.mask.site_ids) / max(1, len(target.substrate.site_ids)),
                            "semantic_blocker": semantic_blocker_for_perturbation(perturbation_type),
                            "repairability_class": repairability_class(perturbation_type),
                        },
                    )
                )
    return rows


def _global_s09_rows(spec: BaselineSpec) -> list[dict[str, Any]]:
    rows = []
    configs = default_scaling_target_configs()
    for config in configs:
        target = config.make_target()
        for task_type in SCALING_TASK_TYPES:
            for seed in DEFAULT_SCALING_SEEDS:
                initial_state, task_metadata = _initial_state_for_task(
                    config=config,
                    target=target,
                    task_type=task_type,
                    seed=int(seed),
                )
                metadata = {
                    "source_research_step_id": "S09",
                    "benchmark_task": "scaling",
                    "task_id": f"S09:{config.config_id}:{task_type}",
                    "simulation_seed": int(seed),
                    "event_cap": S09_EVENT_MULTIPLIER * len(target.substrate.site_ids),
                    "event_multiplier": S09_EVENT_MULTIPLIER,
                    "records_per_run": S09_RECORDS_PER_RUN,
                    "retuning_allowed": False,
                    "large_grid_retuned": False,
                    "tuned_on_scale_label": "small",
                    "scale_label": config.scale_label,
                    "scale_factor": config.scale_factor,
                    "config_id": config.config_id,
                    "source_target_id": config.source_target_id,
                    "task_type": task_type,
                    **task_metadata,
                }
                rows.append(
                    evaluate_baseline_on_state(
                        spec=spec,
                        target=target,
                        initial_state=initial_state,
                        metadata=metadata,
                    )
                )
    return rows


def _global_s10_rows(spec: BaselineSpec) -> list[dict[str, Any]]:
    rows = []
    target = default_symmetry_target()
    for condition in default_initial_conditions():
        for seed in DEFAULT_SYMMETRY_SEEDS:
            initial_state = construct_axis_symmetric_initial_state(target, condition, int(seed))
            rows.append(
                evaluate_baseline_on_state(
                    spec=spec,
                    target=target,
                    initial_state=initial_state,
                    metadata={
                        "source_research_step_id": "S10",
                        "benchmark_task": "symmetry_breaking",
                        "task_id": f"S10:{condition.condition_id}",
                        "simulation_seed": int(seed),
                        "event_cap": S10_EVENT_MULTIPLIER * len(target.substrate.site_ids),
                        "event_multiplier": S10_EVENT_MULTIPLIER,
                        "records_per_run": S10_RECORDS_PER_RUN,
                        "retuning_allowed": False,
                        "condition_id": condition.condition_id,
                        "construction_axis": condition.construction_axis,
                        "symmetry_class": condition.symmetry_class,
                        "near_symmetry_break_swaps": condition.near_symmetry_break_swaps,
                        "repairability_class": "permutation_rearrangement",
                    },
                )
            )
    return rows


def _local_s07_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.to_dict(orient="records"):
        site_count = int(row["site_count"])
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S07",
                "benchmark_task": "scrambled_embryo",
                "task_id": f"S07:{row['target_id']}",
                "target_id": row["target_id"],
                "target_kind": row["target_kind"],
                "target_hash": row["target_hash"],
                "substrate_kind": row["substrate_kind"],
                "site_count": site_count,
                "policy_id": row["policy_id"],
                "policy_family": row["policy_family"],
                "control_class": "local",
                "information_access_tier": LOCAL_INFORMATION_ACCESS_TIER,
                "declared_information_access": row["declared_information_access"],
                "baseline_description": "S07 local control imported without changing information access.",
                "global_state_access": False,
                "global_target_access": False,
                "organizer_cue_added": False,
                "nonlocal_relocation_allowed": False,
                "unfreeze_allowed": False,
                "birth_death_allowed": False,
                "identity_conversion_allowed": False,
                "local_policy_information_access_changed": False,
                "simulation_seed": int(row["simulation_seed"]),
                "event_cap": int(row["event_cap"]),
                "event_multiplier": S07_EVENT_MULTIPLIER,
                "records_per_run": S07_RECORDS_PER_RUN,
                "retuning_allowed": False,
                "large_grid_retuned": False,
                "tuned_on_scale_label": "",
                "initial_target_error": float(row["initial_target_error"]),
                "final_target_error": float(row["final_target_error"]),
                "delta_target_error": float(row["delta_target_error"]),
                "target_recovery_fraction": float(row["target_recovery_fraction"]),
                "initial_aggregate_morphospace_error": float(row["initial_aggregate_morphospace_error"]),
                "final_aggregate_morphospace_error": float(row["final_aggregate_morphospace_error"]),
                "delta_aggregate_morphospace_error": float(row["delta_aggregate_morphospace_error"]),
                "aggregate_recovery_fraction": float(row["aggregate_recovery_fraction"]),
                "initial_displacement_fraction": float(row["initial_displacement_fraction"]),
                "final_displacement_fraction": float(row["final_displacement_fraction"]),
                "accepted_swaps": int(row["accepted_swaps"]),
                "attempted_swaps": int(row["attempted_swaps"]),
                "rejected_actions": int(row["rejected_actions"]),
                "wait_actions": int(row["wait_actions"]),
                "nonlocal_relocations": 0,
                "global_assignment_distance": 0.0,
                "identity_conversion_count": 0,
                "birth_count": 0,
                "death_count": 0,
                "unfreeze_count": 0,
                "total_energy_cost": float(row["total_energy_cost"]),
                "energy_per_site": float(row["total_energy_cost"] / max(1, site_count)),
                "energy_cost_model": "S04 local swap/wait energy charges",
                "population_initial": int(row["population_initial"]),
                "population_final": int(row["population_final"]),
                "population_min": int(row["population_min"]),
                "population_max": int(row["population_max"]),
                "population_delta_total": int(row["population_delta_total"]),
                "identity_multiset_matches_target_initial": True,
                "identity_multiset_matches_target_final": True,
                "stuck_site_count_initial": 0,
                "stuck_site_count_final": 0,
                "empty_site_count_initial": 0,
                "empty_site_count_final": 0,
                "baseline_blocked": False,
                "blocked_reason": "",
                "semantic_blocker": "",
                "repairability_class": "permutation_rearrangement",
                "scale_label": "",
                "scale_factor": np.nan,
                "config_id": "",
                "source_target_id": "",
                "task_type": "",
                "perturbation_type": "full_scramble",
                "mask_id": "",
                "mask_area": 0,
                "mask_area_fraction": 0.0,
                "condition_id": "",
                "construction_axis": "",
                "symmetry_class": "",
                "near_symmetry_break_swaps": 0,
                "source_row_count": 1,
            }
        )
    return pd.DataFrame(rows)


def _local_s08_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.to_dict(orient="records"):
        site_count = int(row["site_count"])
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S08",
                "benchmark_task": "regeneration",
                "task_id": f"S08:{row['target_id']}:{row['perturbation_type']}",
                "target_id": row["target_id"],
                "target_kind": row["target_kind"],
                "target_hash": row["target_hash"],
                "substrate_kind": row["substrate_kind"],
                "site_count": site_count,
                "policy_id": row["policy_id"],
                "policy_family": row["policy_family"],
                "control_class": "local",
                "information_access_tier": LOCAL_INFORMATION_ACCESS_TIER,
                "declared_information_access": row["declared_information_access"],
                "baseline_description": "S08 local swap/wait control imported without changing information access.",
                "global_state_access": False,
                "global_target_access": False,
                "organizer_cue_added": False,
                "nonlocal_relocation_allowed": False,
                "unfreeze_allowed": False,
                "birth_death_allowed": False,
                "identity_conversion_allowed": False,
                "local_policy_information_access_changed": False,
                "simulation_seed": int(row["simulation_seed"]),
                "event_cap": int(row["event_cap"]),
                "event_multiplier": 50,
                "records_per_run": 25,
                "retuning_allowed": False,
                "large_grid_retuned": False,
                "tuned_on_scale_label": "",
                "initial_target_error": float(row["pre_repair_target_error"]),
                "final_target_error": float(row["post_repair_target_error"]),
                "delta_target_error": float(row["delta_target_error"]),
                "target_recovery_fraction": float(row["target_recovery_fraction"]),
                "initial_aggregate_morphospace_error": float(row["pre_repair_aggregate_morphospace_error"]),
                "final_aggregate_morphospace_error": float(row["post_repair_aggregate_morphospace_error"]),
                "delta_aggregate_morphospace_error": float(row["delta_aggregate_morphospace_error"]),
                "aggregate_recovery_fraction": float(row["aggregate_recovery_fraction"]),
                "initial_displacement_fraction": np.nan,
                "final_displacement_fraction": np.nan,
                "accepted_swaps": int(row["accepted_swaps"]),
                "attempted_swaps": int(row["attempted_swaps"]),
                "rejected_actions": int(row["rejected_actions"]),
                "wait_actions": int(row["wait_actions"]),
                "nonlocal_relocations": 0,
                "global_assignment_distance": 0.0,
                "identity_conversion_count": 0,
                "birth_count": 0,
                "death_count": 0,
                "unfreeze_count": 0,
                "total_energy_cost": float(row["total_energy_cost"]),
                "energy_per_site": float(row["total_energy_cost"] / max(1, site_count)),
                "energy_cost_model": "S04 local swap/wait energy charges",
                "population_initial": int(row["population_pre_repair"]),
                "population_final": int(row["population_post_repair"]),
                "population_min": int(row["population_min"]),
                "population_max": int(row["population_max"]),
                "population_delta_total": int(row["population_post_repair"] - row["population_pre_repair"]),
                "identity_multiset_matches_target_initial": bool(row["identity_multiset_matches_target_pre"]),
                "identity_multiset_matches_target_final": bool(row["identity_multiset_matches_target_post"]),
                "stuck_site_count_initial": int(row["stuck_site_count_pre"]),
                "stuck_site_count_final": int(row["stuck_site_count_post"]),
                "empty_site_count_initial": int(row["empty_site_count_pre"]),
                "empty_site_count_final": int(row["empty_site_count_post"]),
                "baseline_blocked": False,
                "blocked_reason": "",
                "semantic_blocker": row["semantic_blocker"],
                "repairability_class": row["repairability_class"],
                "scale_label": "",
                "scale_factor": np.nan,
                "config_id": "",
                "source_target_id": "",
                "task_type": "",
                "perturbation_type": row["perturbation_type"],
                "mask_id": row["mask_id"],
                "mask_area": int(row["mask_area"]),
                "mask_area_fraction": float(row["mask_area"] / max(1, site_count)),
                "condition_id": "",
                "construction_axis": "",
                "symmetry_class": "",
                "near_symmetry_break_swaps": 0,
                "source_row_count": 1,
            }
        )
    return pd.DataFrame(rows)


def _local_s09_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.to_dict(orient="records"):
        site_count = int(row["site_count"])
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S09",
                "benchmark_task": "scaling",
                "task_id": f"S09:{row['config_id']}:{row['task_type']}",
                "target_id": row["target_id"],
                "target_kind": row["target_kind"],
                "target_hash": row["target_hash"],
                "substrate_kind": row["substrate_kind"],
                "site_count": site_count,
                "policy_id": row["policy_id"],
                "policy_family": row["policy_family"],
                "control_class": "local",
                "information_access_tier": LOCAL_INFORMATION_ACCESS_TIER,
                "declared_information_access": row["declared_information_access"],
                "baseline_description": "S09 no-retuning local scaling row imported without changing information access.",
                "global_state_access": False,
                "global_target_access": False,
                "organizer_cue_added": False,
                "nonlocal_relocation_allowed": False,
                "unfreeze_allowed": False,
                "birth_death_allowed": False,
                "identity_conversion_allowed": False,
                "local_policy_information_access_changed": False,
                "simulation_seed": int(row["simulation_seed"]),
                "event_cap": int(row["event_cap"]),
                "event_multiplier": int(row["event_multiplier"]),
                "records_per_run": int(row["records_per_run"]),
                "retuning_allowed": bool(row["retuning_allowed"]),
                "large_grid_retuned": bool(row["large_grid_retuned"]),
                "tuned_on_scale_label": row["tuned_on_scale_label"],
                "initial_target_error": float(row["initial_target_error"]),
                "final_target_error": float(row["final_target_error"]),
                "delta_target_error": float(row["delta_target_error"]),
                "target_recovery_fraction": float(row["target_recovery_fraction"]),
                "initial_aggregate_morphospace_error": float(row["initial_aggregate_morphospace_error"]),
                "final_aggregate_morphospace_error": float(row["final_aggregate_morphospace_error"]),
                "delta_aggregate_morphospace_error": float(row["delta_aggregate_morphospace_error"]),
                "aggregate_recovery_fraction": float(row["aggregate_recovery_fraction"]),
                "initial_displacement_fraction": float(row["initial_displacement_fraction"]),
                "final_displacement_fraction": float(row["final_displacement_fraction"]),
                "accepted_swaps": int(row["accepted_swaps"]),
                "attempted_swaps": int(row["attempted_swaps"]),
                "rejected_actions": int(row["rejected_actions"]),
                "wait_actions": int(row["wait_actions"]),
                "nonlocal_relocations": 0,
                "global_assignment_distance": 0.0,
                "identity_conversion_count": 0,
                "birth_count": 0,
                "death_count": 0,
                "unfreeze_count": 0,
                "total_energy_cost": float(row["total_energy_cost"]),
                "energy_per_site": float(row["energy_per_site"]),
                "energy_cost_model": "S04-compatible local swap/wait energy charges",
                "population_initial": int(row["population_initial"]),
                "population_final": int(row["population_final"]),
                "population_min": int(row["population_min"]),
                "population_max": int(row["population_max"]),
                "population_delta_total": int(row["population_delta_total"]),
                "identity_multiset_matches_target_initial": True,
                "identity_multiset_matches_target_final": True,
                "stuck_site_count_initial": 0,
                "stuck_site_count_final": 0,
                "empty_site_count_initial": 0,
                "empty_site_count_final": 0,
                "baseline_blocked": False,
                "blocked_reason": "",
                "semantic_blocker": row.get("semantic_blocker", ""),
                "repairability_class": row.get("repairability_class", ""),
                "scale_label": row["scale_label"],
                "scale_factor": float(row["scale_factor"]),
                "config_id": row["config_id"],
                "source_target_id": row["source_target_id"],
                "task_type": row["task_type"],
                "perturbation_type": row["perturbation_type"],
                "mask_id": row.get("mask_id", ""),
                "mask_area": int(row.get("mask_area", 0)),
                "mask_area_fraction": float(row.get("mask_area_fraction", 0.0)),
                "condition_id": "",
                "construction_axis": "",
                "symmetry_class": "",
                "near_symmetry_break_swaps": 0,
                "source_row_count": 1,
            }
        )
    return pd.DataFrame(rows)


def _local_s10_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.to_dict(orient="records"):
        site_count = int(row["site_count"])
        payload = {
            "research_step_id": STEP_ID,
            "source_research_step_id": "S10",
            "benchmark_task": "symmetry_breaking",
            "task_id": f"S10:{row['condition_id']}",
            "target_id": row["target_id"],
            "target_kind": row["target_kind"],
            "target_hash": row["target_hash"],
            "substrate_kind": row["substrate_kind"],
            "site_count": site_count,
            "policy_id": row["policy_id"],
            "policy_family": row["policy_family"],
            "control_class": "local",
            "information_access_tier": LOCAL_INFORMATION_ACCESS_TIER,
            "declared_information_access": row["declared_information_access"],
            "baseline_description": "S10 local symmetry-breaking row imported without adding organizer or global-axis cues.",
            "global_state_access": False,
            "global_target_access": False,
            "organizer_cue_added": bool(row["organizer_cue_added"]),
            "nonlocal_relocation_allowed": False,
            "unfreeze_allowed": False,
            "birth_death_allowed": False,
            "identity_conversion_allowed": False,
            "local_policy_information_access_changed": False,
            "simulation_seed": int(row["simulation_seed"]),
            "event_cap": int(row["event_cap"]),
            "event_multiplier": int(row["event_multiplier"]),
            "records_per_run": int(row["records_per_run"]),
            "retuning_allowed": False,
            "large_grid_retuned": False,
            "tuned_on_scale_label": "",
            "initial_target_error": float(row["initial_target_error"]),
            "final_target_error": float(row["final_target_error"]),
            "delta_target_error": float(row["delta_target_error"]),
            "target_recovery_fraction": float(row["target_recovery_fraction"]),
            "initial_aggregate_morphospace_error": float(row["initial_aggregate_morphospace_error"]),
            "final_aggregate_morphospace_error": float(row["final_aggregate_morphospace_error"]),
            "delta_aggregate_morphospace_error": float(row["delta_aggregate_morphospace_error"]),
            "aggregate_recovery_fraction": float(row["aggregate_recovery_fraction"]),
            "initial_displacement_fraction": float(row["initial_displacement_fraction"]),
            "final_displacement_fraction": float(row["final_displacement_fraction"]),
            "accepted_swaps": int(row["accepted_swaps"]),
            "attempted_swaps": int(row["attempted_swaps"]),
            "rejected_actions": int(row["rejected_actions"]),
            "wait_actions": int(row["wait_actions"]),
            "nonlocal_relocations": 0,
            "global_assignment_distance": 0.0,
            "identity_conversion_count": 0,
            "birth_count": 0,
            "death_count": 0,
            "unfreeze_count": 0,
            "total_energy_cost": float(row["total_energy_cost"]),
            "energy_per_site": float(row["energy_per_site"]),
            "energy_cost_model": "S04-compatible local swap/wait energy charges",
            "population_initial": int(row["population_initial"]),
            "population_final": int(row["population_final"]),
            "population_min": int(row["population_min"]),
            "population_max": int(row["population_max"]),
            "population_delta_total": int(row["population_delta_total"]),
            "identity_multiset_matches_target_initial": True,
            "identity_multiset_matches_target_final": True,
            "stuck_site_count_initial": 0,
            "stuck_site_count_final": 0,
            "empty_site_count_initial": 0,
            "empty_site_count_final": 0,
            "baseline_blocked": False,
            "blocked_reason": "",
            "semantic_blocker": "",
            "repairability_class": "permutation_rearrangement",
            "scale_label": "",
            "scale_factor": np.nan,
            "config_id": "",
            "source_target_id": "",
            "task_type": "",
            "perturbation_type": "",
            "mask_id": "",
            "mask_area": 0,
            "mask_area_fraction": 0.0,
            "condition_id": row["condition_id"],
            "construction_axis": row["construction_axis"],
            "symmetry_class": row["symmetry_class"],
            "near_symmetry_break_swaps": int(row["near_symmetry_break_swaps"]),
            "source_row_count": 1,
            "initial_chosen_axis": row["initial_chosen_axis"],
            "final_chosen_axis": row["final_chosen_axis"],
            "initial_axis_margin": float(row["initial_axis_margin"]),
            "final_axis_margin": float(row["final_axis_margin"]),
            "canonical_axis_selected_initial": bool(row["canonical_axis_selected_initial"]),
            "canonical_axis_selected_final": bool(row["canonical_axis_selected_final"]),
            "construction_axis_retained_final": bool(row["construction_axis_retained_final"]),
            "axis_switched_to_canonical": bool(row["axis_switched_to_canonical"]),
            "axis_choice_outcome": row["axis_choice_outcome"],
        }
        rows.append(payload)
    return pd.DataFrame(rows)


def _axis_summary_fields(
    target: TargetMorphology,
    initial_state: SubstrateState,
    final_state: SubstrateState,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    initial_axis = axis_choice_metrics(target, initial_state, "initial")
    final_axis = axis_choice_metrics(target, final_state, "final")
    construction_axis = str(metadata.get("construction_axis", ""))
    return {
        "initial_chosen_axis": initial_axis["chosen_axis"],
        "final_chosen_axis": final_axis["chosen_axis"],
        "initial_axis_margin": float(initial_axis["axis_margin"]),
        "final_axis_margin": float(final_axis["axis_margin"]),
        "canonical_axis_selected_initial": initial_axis["chosen_axis"] == "vertical",
        "canonical_axis_selected_final": final_axis["chosen_axis"] == "vertical",
        "construction_axis_retained_final": final_axis["chosen_axis"] == construction_axis,
        "axis_switched_to_canonical": construction_axis != "vertical" and final_axis["chosen_axis"] == "vertical",
        "axis_choice_outcome": _axis_choice_outcome(construction_axis, str(final_axis["chosen_axis"])),
    }


def _axis_choice_outcome(construction_axis: str, final_axis: str) -> str:
    if final_axis == "ambiguous":
        return "ambiguous_or_unresolved"
    if final_axis == "vertical":
        return "canonical_vertical_selected"
    if final_axis == construction_axis:
        return "construction_axis_retained"
    return "noncanonical_axis_selected"


def _validation_row(
    validation_case: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "validation_case": validation_case,
        "success": bool(success),
        "expected_json": json.dumps(expected, sort_keys=True, separators=(",", ":"), default=str),
        "observed_json": json.dumps(observed, sort_keys=True, separators=(",", ":"), default=str),
        "detail": detail,
    }


def _benchmark_task_from_s12(source_step: str, source_task_type: str) -> str:
    if source_step == "S07":
        return "scrambled_embryo"
    if source_step == "S08":
        return "regeneration"
    if source_step == "S09":
        return "scaling"
    if source_step == "S10":
        return "symmetry_breaking"
    if source_step == "S11":
        return source_task_type
    return source_task_type


def _s13_task_id_from_s12(source_step: str, task_id: str, row: Mapping[str, Any]) -> str:
    if source_step == "S07":
        return f"S07:{row['target_id']}"
    if source_step == "S08":
        return task_id
    if source_step == "S09":
        parts = task_id.split(":", 2)
        if len(parts) == 3:
            return f"S09:{row['target_id'].replace('_small', ':small').replace('_large', ':large')}:{parts[1]}"
        return task_id
    if source_step == "S10":
        return task_id
    return task_id


def _identity_edit_count(target: TargetMorphology, state: SubstrateState) -> int:
    edits = 0
    for site_id in target.substrate.site_ids:
        cell = state.cell_at(site_id)
        if cell is None:
            edits += 1
            continue
        try:
            identity = identity_from_substrate_cell(cell, fallback_scalar=False)
        except ValueError:
            edits += 1
            continue
        if target.schema.distance(identity, target.identities_by_site[int(site_id)]) > 1e-12:
            edits += 1
    return int(edits)


def _occupied_count(state: SubstrateState) -> int:
    return sum(1 for site_id in state.site_ids if state.cell_at(site_id) is not None)


def _stuck_count(state: SubstrateState) -> int:
    return sum(1 for site_id in state.site_ids if _is_stuck(state.cell_at(site_id)))


def _is_stuck(cell: SubstrateCell | None) -> bool:
    return cell is not None and str(cell.status).strip().lower() in {STUCK_STATUS, "stuck", "freeze", "frozen"}


def _with_active_status(cell: SubstrateCell) -> SubstrateCell:
    metadata = dict(cell.metadata)
    metadata["s13_unfrozen_by_global_baseline"] = True
    return SubstrateCell(cell_id=cell.cell_id, value=cell.value, label=cell.label, status="active", metadata=metadata)


def _identity_signature(identity: CellIdentity) -> str:
    return json.dumps(identity.compact_dict(), sort_keys=True, separators=(",", ":"), default=str)


def _coordinate_distance(substrate: SubstrateState, source_site_id: int, target_site_id: int) -> float:
    left = substrate.coordinate(int(source_site_id))
    right = substrate.coordinate(int(target_site_id))
    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=False)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return float("nan")
    return float(float(numerator) / denominator)
