"""Regeneration perturbation benchmark helpers for E05 S08."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.e05.actions import ActionExecutor
from src.e05.cell_identity import CellIdentity, attach_identity, identity_from_substrate_cell
from src.e05.scrambled_embryo import (
    RUNNABLE_POLICY_IDS,
    clone_substrate_state,
    compute_state_metrics,
    policy_by_id,
    recovery_fraction,
    stable_int_hash,
    two_dimensional_targets,
)
from src.e05.substrates import SubstrateCell, SubstrateState
from src.e05.targets import TargetMorphology, default_target_gallery


PERTURBATION_TYPES = ("missing_patch", "frozen_patch", "rotated_patch", "duplicated_patch", "foreign_patch")
DEFAULT_REGENERATION_SEEDS = (2101, 2102, 2103)
DEFAULT_EVENT_MULTIPLIER = 50
DEFAULT_RECORDS_PER_RUN = 25
STUCK_STATUS = "stuck"


@dataclass(frozen=True)
class PatchMask:
    """One deterministic rectangular perturbation mask."""

    mask_id: str
    site_ids: tuple[int, ...]
    coordinates: tuple[tuple[int, int], ...]
    origin: tuple[int, int]
    width: int
    height: int
    seed: int

    def compact_dict(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "site_ids": list(self.site_ids),
            "coordinates": [list(coord) for coord in self.coordinates],
            "origin": list(self.origin),
            "width": int(self.width),
            "height": int(self.height),
            "seed": int(self.seed),
            "mask_area": len(self.site_ids),
        }


@dataclass(frozen=True)
class PerturbedState:
    """State plus provenance for one S08 perturbation."""

    target: TargetMorphology
    perturbation_type: str
    seed: int
    mask: PatchMask
    state: SubstrateState
    baseline_target_error: float
    perturbed_target_error: float
    perturbed_aggregate_morphospace_error: float
    population_baseline: int
    population_perturbed: int
    identity_multiset_matches_target: bool
    stuck_site_count: int
    empty_site_count: int
    source_site_ids: tuple[int, ...] = ()
    donor_target_id: str | None = None
    semantic_blocker: str = ""

    def mask_row(self) -> dict[str, Any]:
        payload = {
            "research_step_id": "S08",
            "target_id": self.target.target_id,
            "target_kind": self.target.target_kind,
            "perturbation_type": self.perturbation_type,
            "simulation_seed": int(self.seed),
            "mask_id": self.mask.mask_id,
            "mask_site_ids_json": json.dumps(list(self.mask.site_ids), separators=(",", ":")),
            "mask_coordinates_json": json.dumps([list(coord) for coord in self.mask.coordinates], separators=(",", ":")),
            "mask_area": len(self.mask.site_ids),
            "mask_origin_x": self.mask.origin[0],
            "mask_origin_y": self.mask.origin[1],
            "mask_width": self.mask.width,
            "mask_height": self.mask.height,
            "source_site_ids_json": json.dumps(list(self.source_site_ids), separators=(",", ":")),
            "donor_target_id": self.donor_target_id or "",
            "semantic_blocker": self.semantic_blocker,
        }
        return payload


@dataclass(frozen=True)
class RegenerationRunResult:
    """One regeneration trajectory and summary."""

    summary_row: dict[str, Any]
    trace_rows: tuple[dict[str, Any], ...]
    metric_rows: tuple[dict[str, Any], ...]
    mask_row: dict[str, Any]
    perturbed_state: SubstrateState
    final_state: SubstrateState


def select_patch_mask(target: TargetMorphology, perturbation_type: str, seed: int) -> PatchMask:
    """Select a compact deterministic 2D patch for a target and seed."""

    if perturbation_type not in PERTURBATION_TYPES:
        raise ValueError(f"unsupported perturbation_type: {perturbation_type}")
    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    height = int(target.substrate.dimensions.get("height", 1))
    if width <= 1 or height <= 1:
        raise ValueError("S08 patch masks require a 2D target")
    patch_width = min(3, width)
    patch_height = min(3, height)
    if height <= 4:
        patch_height = min(2, height)
    rng = random.Random(int(seed) + stable_int_hash(f"{target.target_id}:{perturbation_type}:mask"))
    center_x = max(0, (width - patch_width) // 2)
    center_y = max(0, (height - patch_height) // 2)
    jitter_x = rng.choice((-1, 0, 1)) if width > patch_width else 0
    jitter_y = rng.choice((-1, 0, 1)) if height > patch_height else 0
    origin_x = min(max(0, center_x + jitter_x), width - patch_width)
    origin_y = min(max(0, center_y + jitter_y), height - patch_height)
    coordinates = tuple(
        (x, y)
        for y in range(origin_y, origin_y + patch_height)
        for x in range(origin_x, origin_x + patch_width)
    )
    site_ids = tuple(y * width + x for x, y in coordinates)
    mask_id = f"{target.target_id}:{perturbation_type}:seed_{int(seed)}:{origin_x}_{origin_y}_{patch_width}x{patch_height}"
    return PatchMask(
        mask_id=mask_id,
        site_ids=site_ids,
        coordinates=coordinates,
        origin=(origin_x, origin_y),
        width=patch_width,
        height=patch_height,
        seed=int(seed),
    )


def apply_regeneration_perturbation(
    target: TargetMorphology,
    perturbation_type: str,
    seed: int,
    *,
    donor_targets: Sequence[TargetMorphology] | None = None,
) -> PerturbedState:
    """Apply one S08 perturbation to a constructed S03 target state."""

    mask = select_patch_mask(target, perturbation_type, seed)
    baseline = target.constructed_substrate()
    cells_by_site = {site_id: baseline.cell_at(site_id) for site_id in baseline.site_ids}
    source_site_ids: tuple[int, ...] = ()
    donor_target_id: str | None = None
    semantic_blocker = semantic_blocker_for_perturbation(perturbation_type)

    if perturbation_type == "missing_patch":
        for site_id in mask.site_ids:
            cells_by_site[site_id] = None
    elif perturbation_type == "rotated_patch":
        cells_by_site = _with_rotated_patch(target, cells_by_site, mask, stuck=False)
    elif perturbation_type == "frozen_patch":
        cells_by_site = _with_rotated_patch(target, cells_by_site, mask, stuck=True)
    elif perturbation_type == "duplicated_patch":
        source_site_ids = choose_duplicate_source_sites(target, mask, seed)
        for target_site_id, source_site_id in zip(mask.site_ids, source_site_ids, strict=True):
            source_cell = baseline.cell_at(source_site_id)
            if source_cell is None:
                raise ValueError("duplicate source unexpectedly empty")
            cells_by_site[target_site_id] = duplicate_cell(source_cell, target_site_id, perturbation_type, seed)
    elif perturbation_type == "foreign_patch":
        donor = choose_foreign_donor(target, donor_targets or two_dimensional_targets(default_target_gallery()), seed)
        donor_target_id = donor.target_id
        donor_state = donor.constructed_substrate()
        donor_site_ids = donor.substrate.site_ids
        for idx, target_site_id in enumerate(mask.site_ids):
            donor_cell = donor_state.cell_at(donor_site_ids[idx % len(donor_site_ids)])
            if donor_cell is None:
                raise ValueError("foreign donor unexpectedly empty")
            cells_by_site[target_site_id] = duplicate_cell(donor_cell, target_site_id, perturbation_type, seed, donor_target_id=donor.target_id)
    else:
        raise ValueError(f"unsupported perturbation_type: {perturbation_type}")

    state = target.substrate.copy_empty()
    for site_id in state.site_ids:
        cell = cells_by_site[site_id]
        if cell is not None:
            state.place_cell(site_id, cell)
    perturbed_metrics = compute_state_metrics(target, state, "perturbed")
    return PerturbedState(
        target=target,
        perturbation_type=perturbation_type,
        seed=int(seed),
        mask=mask,
        state=state,
        baseline_target_error=float(target.target_error(baseline)),
        perturbed_target_error=float(perturbed_metrics["target_error"]),
        perturbed_aggregate_morphospace_error=float(perturbed_metrics["aggregate_morphospace_error"]),
        population_baseline=len(target.substrate.site_ids),
        population_perturbed=sum(1 for site_id in state.site_ids if state.cell_at(site_id) is not None),
        identity_multiset_matches_target=identity_multiset_matches_target(target, state),
        stuck_site_count=sum(1 for site_id in state.site_ids if (state.cell_at(site_id) is not None and state.cell_at(site_id).status == STUCK_STATUS)),
        empty_site_count=sum(1 for site_id in state.site_ids if state.cell_at(site_id) is None),
        source_site_ids=source_site_ids,
        donor_target_id=donor_target_id,
        semantic_blocker=semantic_blocker,
    )


def simulate_regeneration_run(
    *,
    target: TargetMorphology,
    perturbation_type: str,
    policy_id: str,
    seed: int,
    donor_targets: Sequence[TargetMorphology] | None = None,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
) -> RegenerationRunResult:
    """Run one S08 perturbation trajectory with S07 local controls."""

    perturbed = apply_regeneration_perturbation(target, perturbation_type, seed, donor_targets=donor_targets)
    state = clone_substrate_state(perturbed.state)
    policy = policy_by_id(policy_id)
    executor = ActionExecutor()
    rng = random.Random(int(seed) * 65_537 + stable_int_hash(f"{target.target_id}:{perturbation_type}:{policy_id}"))
    site_ids = tuple(target.substrate.site_ids)
    event_cap = max(1, int(event_multiplier) * len(site_ids))
    record_every = max(1, event_cap // max(1, int(records_per_run)))
    initial_metrics = compute_state_metrics(target, state, "perturbed")
    trace_rows = [
        _trace_row(
            target=target,
            perturbation_type=perturbation_type,
            mask_id=perturbed.mask.mask_id,
            policy_id=policy.policy_id,
            policy_family=policy.policy_family,
            seed=seed,
            event_step=0,
            accepted_swaps=0,
            attempted_swaps=0,
            rejected_actions=0,
            wait_actions=0,
            energy_cost=0.0,
            population_count=perturbed.population_perturbed,
            target_error=initial_metrics["target_error"],
            aggregate_morphospace_error=initial_metrics["aggregate_morphospace_error"],
        )
    ]
    metric_rows = _tag_metric_rows(
        initial_metrics["metric_rows"],
        target=target,
        perturbation_type=perturbation_type,
        mask_id=perturbed.mask.mask_id,
        policy_id=policy.policy_id,
        policy_family=policy.policy_family,
        seed=seed,
        state_label="perturbed",
    )

    accepted_swaps = 0
    attempted_swaps = 0
    rejected_actions = 0
    wait_actions = 0
    total_energy = 0.0
    population_min = perturbed.population_perturbed
    population_max = perturbed.population_perturbed
    final_metrics = initial_metrics
    for event_step in range(1, event_cap + 1):
        occupied_site_ids = tuple(site_id for site_id in site_ids if state.cell_at(site_id) is not None)
        if not occupied_site_ids:
            break
        actor_site_id = rng.choice(occupied_site_ids)
        request = policy.propose(state=state, target=target, actor_site_id=actor_site_id, rng=rng)
        result = executor.execute(state, request)
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
            trace_rows.append(
                _trace_row(
                    target=target,
                    perturbation_type=perturbation_type,
                    mask_id=perturbed.mask.mask_id,
                    policy_id=policy.policy_id,
                    policy_family=policy.policy_family,
                    seed=seed,
                    event_step=event_step,
                    accepted_swaps=accepted_swaps,
                    attempted_swaps=attempted_swaps,
                    rejected_actions=rejected_actions,
                    wait_actions=wait_actions,
                    energy_cost=total_energy,
                    population_count=sum(1 for site_id in site_ids if state.cell_at(site_id) is not None),
                    target_error=final_metrics["target_error"],
                    aggregate_morphospace_error=final_metrics["aggregate_morphospace_error"],
                )
            )
    final_metrics = compute_state_metrics(target, state, "final")
    metric_rows.extend(
        _tag_metric_rows(
            final_metrics["metric_rows"],
            target=target,
            perturbation_type=perturbation_type,
            mask_id=perturbed.mask.mask_id,
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
                perturbation_type=perturbation_type,
                mask_id=perturbed.mask.mask_id,
                policy_id=policy.policy_id,
                policy_family=policy.policy_family,
                seed=seed,
                event_step=event_cap,
                accepted_swaps=accepted_swaps,
                attempted_swaps=attempted_swaps,
                rejected_actions=rejected_actions,
                wait_actions=wait_actions,
                energy_cost=total_energy,
                population_count=sum(1 for site_id in site_ids if state.cell_at(site_id) is not None),
                target_error=final_metrics["target_error"],
                aggregate_morphospace_error=final_metrics["aggregate_morphospace_error"],
            )
        )
    else:
        trace_rows[-1]["target_error"] = final_metrics["target_error"]
        trace_rows[-1]["aggregate_morphospace_error"] = final_metrics["aggregate_morphospace_error"]

    final_population = sum(1 for site_id in site_ids if state.cell_at(site_id) is not None)
    summary_row = {
        "research_step_id": "S08",
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "target_hash": target.target_hash,
        "substrate_kind": target.substrate.kind,
        "site_count": len(site_ids),
        "perturbation_type": perturbation_type,
        "mask_id": perturbed.mask.mask_id,
        "mask_area": len(perturbed.mask.site_ids),
        "policy_id": policy.policy_id,
        "policy_family": policy.policy_family,
        "declared_information_access": policy.declared_information_access,
        "simulation_seed": int(seed),
        "event_cap": int(event_cap),
        "record_every": int(record_every),
        "baseline_target_error": perturbed.baseline_target_error,
        "pre_repair_target_error": float(initial_metrics["target_error"]),
        "post_repair_target_error": float(final_metrics["target_error"]),
        "delta_target_error": float(final_metrics["target_error"] - initial_metrics["target_error"]),
        "target_recovery_fraction": recovery_fraction(float(initial_metrics["target_error"]), float(final_metrics["target_error"])),
        "pre_repair_aggregate_morphospace_error": float(initial_metrics["aggregate_morphospace_error"]),
        "post_repair_aggregate_morphospace_error": float(final_metrics["aggregate_morphospace_error"]),
        "delta_aggregate_morphospace_error": float(final_metrics["aggregate_morphospace_error"] - initial_metrics["aggregate_morphospace_error"]),
        "aggregate_recovery_fraction": recovery_fraction(
            float(initial_metrics["aggregate_morphospace_error"]),
            float(final_metrics["aggregate_morphospace_error"]),
        ),
        "population_baseline": int(perturbed.population_baseline),
        "population_pre_repair": int(perturbed.population_perturbed),
        "population_post_repair": int(final_population),
        "population_min": int(population_min),
        "population_max": int(population_max),
        "population_delta_from_baseline": int(final_population - perturbed.population_baseline),
        "identity_multiset_matches_target_pre": bool(perturbed.identity_multiset_matches_target),
        "identity_multiset_matches_target_post": bool(identity_multiset_matches_target(target, state)),
        "empty_site_count_pre": int(perturbed.empty_site_count),
        "empty_site_count_post": sum(1 for site_id in site_ids if state.cell_at(site_id) is None),
        "stuck_site_count_pre": int(perturbed.stuck_site_count),
        "stuck_site_count_post": sum(1 for site_id in site_ids if (state.cell_at(site_id) is not None and state.cell_at(site_id).status == STUCK_STATUS)),
        "semantic_blocker": perturbed.semantic_blocker,
        "repairability_class": repairability_class(perturbation_type),
        "extra_repair_semantics_required": bool(perturbed.semantic_blocker),
        "birth_death_or_identity_conversion_required": perturbation_type in {"missing_patch", "duplicated_patch", "foreign_patch"},
        "unfreeze_required": perturbation_type == "frozen_patch",
        "policy_overextension_avoided": bool(perturbed.semantic_blocker),
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "wait_actions": int(wait_actions),
        "total_energy_cost": float(total_energy),
    }
    return RegenerationRunResult(
        summary_row=summary_row,
        trace_rows=tuple(trace_rows),
        metric_rows=tuple(metric_rows),
        mask_row=perturbed.mask_row(),
        perturbed_state=perturbed.state,
        final_state=clone_substrate_state(state),
    )


def run_regeneration_benchmark(
    *,
    targets: Sequence[TargetMorphology] | None = None,
    perturbation_types: Sequence[str] = PERTURBATION_TYPES,
    policy_ids: Sequence[str] = RUNNABLE_POLICY_IDS,
    seeds: Sequence[int] = DEFAULT_REGENERATION_SEEDS,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str, int], RegenerationRunResult]]:
    """Run the compact S08 target by perturbation by policy matrix."""

    selected_targets = two_dimensional_targets(targets or default_target_gallery())
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    mask_rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    run_results: dict[tuple[str, str, str, int], RegenerationRunResult] = {}
    donor_pool = selected_targets if len(selected_targets) > 1 else two_dimensional_targets(default_target_gallery())
    for target in selected_targets:
        for perturbation_type in perturbation_types:
            if perturbation_type not in PERTURBATION_TYPES:
                raise ValueError(f"unsupported perturbation_type: {perturbation_type}")
            for policy_id in policy_ids:
                for seed in seeds:
                    result = simulate_regeneration_run(
                        target=target,
                        perturbation_type=perturbation_type,
                        policy_id=policy_id,
                        seed=int(seed),
                        donor_targets=donor_pool,
                        event_multiplier=event_multiplier,
                        records_per_run=records_per_run,
                    )
                    summary_rows.append(result.summary_row)
                    trace_rows.extend(result.trace_rows)
                    metric_rows.extend(result.metric_rows)
                    mask_rows_by_key[(target.target_id, perturbation_type, int(seed))] = result.mask_row
                    run_results[(target.target_id, perturbation_type, policy_id, int(seed))] = result
    return summary_rows, trace_rows, metric_rows, list(mask_rows_by_key.values()), run_results


def choose_duplicate_source_sites(target: TargetMorphology, mask: PatchMask, seed: int) -> tuple[int, ...]:
    """Choose a same-shaped source patch outside the perturbation mask when possible."""

    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    height = int(target.substrate.dimensions.get("height", 1))
    mask_set = set(mask.site_ids)
    origins = []
    for y in range(0, height - mask.height + 1):
        for x in range(0, width - mask.width + 1):
            candidate = tuple((y + dy) * width + (x + dx) for dy in range(mask.height) for dx in range(mask.width))
            if not mask_set.intersection(candidate):
                origins.append((x, y, candidate))
    rng = random.Random(int(seed) + stable_int_hash(f"{target.target_id}:duplicate_source"))
    if origins:
        _x, _y, candidate = rng.choice(origins)
        return tuple(candidate)
    outside = [site_id for site_id in target.substrate.site_ids if site_id not in mask_set]
    if len(outside) < len(mask.site_ids):
        raise ValueError("not enough outside sites to duplicate a patch")
    rng.shuffle(outside)
    return tuple(outside[: len(mask.site_ids)])


def choose_foreign_donor(target: TargetMorphology, targets: Sequence[TargetMorphology], seed: int) -> TargetMorphology:
    candidates = [item for item in targets if item.target_id != target.target_id and item.schema.schema_id == target.schema.schema_id]
    if not candidates:
        raise ValueError(f"no compatible foreign donor target for {target.target_id}")
    rng = random.Random(int(seed) + stable_int_hash(f"{target.target_id}:foreign_donor"))
    return rng.choice(candidates)


def duplicate_cell(
    source_cell: SubstrateCell,
    target_site_id: int,
    perturbation_type: str,
    seed: int,
    *,
    donor_target_id: str | None = None,
) -> SubstrateCell:
    identity = identity_from_substrate_cell(source_cell, fallback_scalar=False)
    metadata = {
        **identity.metadata,
        "s08_perturbation_type": perturbation_type,
        "s08_source_cell_id": source_cell.cell_id,
        "s08_seed": int(seed),
    }
    if donor_target_id is not None:
        metadata["s08_donor_target_id"] = donor_target_id
    copied_identity = CellIdentity(
        identity_id=identity.identity_id,
        components=dict(identity.components),
        target_preferences=identity.target_preferences,
        metadata=metadata,
    )
    return attach_identity(
        SubstrateCell(
            cell_id=f"s08_{perturbation_type}_{int(seed)}_site_{int(target_site_id)}_from_{source_cell.cell_id}",
            value=source_cell.value,
            label=source_cell.label,
            status="active",
            metadata={key: value for key, value in source_cell.metadata.items() if key != "e05_identity"},
        ),
        copied_identity,
    )


def semantic_blocker_for_perturbation(perturbation_type: str) -> str:
    if perturbation_type == "missing_patch":
        return (
            "Full repair requires birth/division or an equivalent source of replacement identities; S08 keeps swap-only S07 controls "
            "and does not synthesize missing cells."
        )
    if perturbation_type == "frozen_patch":
        return (
            "Full repair requires an explicit unfreeze or repair action for stuck cells; S08 does not reinterpret swap, crawl, divide, or die as unfreeze."
        )
    if perturbation_type == "duplicated_patch":
        return (
            "Full repair requires removing duplicate identities and restoring missing originals, which needs death plus birth or identity conversion semantics."
        )
    if perturbation_type == "foreign_patch":
        return (
            "Full repair requires removing or converting foreign identities; S08 does not add identity conversion or death/birth policy semantics."
        )
    return ""


def repairability_class(perturbation_type: str) -> str:
    if perturbation_type == "rotated_patch":
        return "swap_repairable_rearrangement"
    if perturbation_type == "frozen_patch":
        return "blocked_by_stuck_cells"
    if perturbation_type == "missing_patch":
        return "blocked_by_population_deficit"
    if perturbation_type in {"duplicated_patch", "foreign_patch"}:
        return "blocked_by_identity_multiset_mismatch"
    return "unknown"


def identity_multiset_matches_target(target: TargetMorphology, state: SubstrateState) -> bool:
    observed = Counter()
    for site_id in state.site_ids:
        cell = state.cell_at(site_id)
        if cell is None:
            continue
        try:
            identity = identity_from_substrate_cell(cell, fallback_scalar=False)
        except ValueError:
            continue
        observed[_identity_signature(identity)] += 1
    expected = Counter(_identity_signature(identity) for identity in target.identities_by_site.values())
    return observed == expected


def write_regeneration_trace_zarr(trace_df: Any, output_path: Path) -> None:
    """Write S08 trace rows as a compact zarr group with categorical codes."""

    import pandas as pd
    import zarr

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trace_df).reset_index(drop=True)
    root = zarr.open_group(str(output_path), mode="w")
    root.attrs["schema"] = "eidosoma.e05_s08.regeneration_traces.zarr.v1"
    root.attrs["row_count"] = int(len(df))
    root.attrs["columns"] = list(df.columns)
    category_mappings: dict[str, list[str]] = {}
    for column in df.columns:
        series = df[column]
        if series.dtype == object:
            categories = sorted(series.fillna("").astype(str).unique().tolist())
            mapping = {value: idx for idx, value in enumerate(categories)}
            codes = series.fillna("").astype(str).map(mapping).to_numpy(dtype=np.int32)
            root.create_array(f"{column}_category_code", data=codes, chunks=(max(1, min(len(codes), 1024)),))
            category_mappings[column] = categories
        elif str(series.dtype).startswith("bool"):
            values = series.to_numpy(dtype=np.int8)
            root.create_array(column, data=values, chunks=(max(1, min(len(values), 1024)),))
        elif np.issubdtype(series.dtype, np.integer):
            values = series.to_numpy(dtype=np.int64)
            root.create_array(column, data=values, chunks=(max(1, min(len(values), 1024)),))
        else:
            values = series.to_numpy(dtype=np.float64)
            root.create_array(column, data=values, chunks=(max(1, min(len(values), 1024)),))
    root.attrs["categorical_columns"] = category_mappings


def validate_regeneration_trace_zarr(output_path: Path, expected_rows: int) -> dict[str, Any]:
    import zarr

    if not output_path.exists():
        return {"exists": False, "row_count": 0, "arrays": [], "all_arrays_row_aligned": False}
    root = zarr.open_group(str(output_path), mode="r")
    arrays = sorted(root.array_keys())
    row_count = int(root.attrs.get("row_count", 0))
    aligned = row_count == int(expected_rows) and all(int(root[name].shape[0]) == row_count for name in arrays)
    return {
        "exists": True,
        "row_count": row_count,
        "arrays": arrays,
        "all_arrays_row_aligned": bool(aligned),
        "schema": root.attrs.get("schema"),
    }


def _with_rotated_patch(
    target: TargetMorphology,
    cells_by_site: Mapping[int, SubstrateCell | None],
    mask: PatchMask,
    *,
    stuck: bool,
) -> dict[int, SubstrateCell | None]:
    rotated = dict(cells_by_site)
    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    x0, y0 = mask.origin
    x1 = x0 + mask.width - 1
    y1 = y0 + mask.height - 1
    for site_id, (x, y) in zip(mask.site_ids, mask.coordinates, strict=True):
        source_site_id = (y0 + y1 - y) * width + (x0 + x1 - x)
        source_cell = cells_by_site[source_site_id]
        if source_cell is None:
            rotated[site_id] = None
        elif stuck:
            rotated[site_id] = _with_status(source_cell, STUCK_STATUS)
        else:
            rotated[site_id] = source_cell
    return rotated


def _with_status(cell: SubstrateCell, status: str) -> SubstrateCell:
    metadata = dict(cell.metadata)
    metadata["s08_status_perturbation"] = status
    return SubstrateCell(
        cell_id=cell.cell_id,
        value=cell.value,
        label=cell.label,
        status=status,
        metadata=metadata,
    )


def _identity_signature(identity: CellIdentity) -> str:
    return json.dumps(identity.compact_dict(), sort_keys=True, separators=(",", ":"), default=str)


def _trace_row(
    *,
    target: TargetMorphology,
    perturbation_type: str,
    mask_id: str,
    policy_id: str,
    policy_family: str,
    seed: int,
    event_step: int,
    accepted_swaps: int,
    attempted_swaps: int,
    rejected_actions: int,
    wait_actions: int,
    energy_cost: float,
    population_count: int,
    target_error: float,
    aggregate_morphospace_error: float,
) -> dict[str, Any]:
    return {
        "research_step_id": "S08",
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "perturbation_type": perturbation_type,
        "mask_id": mask_id,
        "policy_id": policy_id,
        "policy_family": policy_family,
        "simulation_seed": int(seed),
        "event_step": int(event_step),
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "wait_actions": int(wait_actions),
        "energy_cost": float(energy_cost),
        "population_count": int(population_count),
        "target_error": float(target_error),
        "aggregate_morphospace_error": float(aggregate_morphospace_error),
    }


def _tag_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: TargetMorphology,
    perturbation_type: str,
    mask_id: str,
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
                "research_step_id": "S08",
                "target_id": target.target_id,
                "target_kind": target.target_kind,
                "perturbation_type": perturbation_type,
                "mask_id": mask_id,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "simulation_seed": int(seed),
                "benchmark_state_label": state_label,
            }
        )
        tagged.append(payload)
    return tagged
