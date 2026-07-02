"""GPU-batched tissue simulation helpers for E05 S11.

The S11 kernel intentionally covers the full-occupancy, square-grid,
swap/wait subset of the E05 benchmark.  That subset includes selected S07
scrambled targets, S08 rotated-patch repair, S09 scaled scramble tests, and
S10 symmetry starts.  Perturbations that need birth, death, identity
conversion, unfreeze semantics, or irregular graph vectorization are reported
as blocked scope rows instead of being silently remapped.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from src.e05.actions import MorphogenesisActionRequest
from src.e05.cell_identity import identity_from_substrate_cell
from src.e05.regeneration import apply_regeneration_perturbation, semantic_blocker_for_perturbation
from src.e05.scaling import (
    cached_adjacent_swap_target_error_delta,
    default_scaling_target_configs,
    execute_swap_wait_fast,
    target_identity_distance_lookup,
)
from src.e05.scrambled_embryo import (
    RUNNABLE_POLICY_IDS,
    clone_substrate_state,
    policy_by_id,
    recovery_fraction,
    scramble_target_state,
    stable_int_hash,
)
from src.e05.substrates import SubstrateState
from src.e05.symmetry_breaking import (
    construct_axis_symmetric_initial_state,
    default_initial_conditions,
    default_symmetry_target,
)
from src.e05.targets import (
    TargetMorphology,
    boundary_target,
    gradient_target,
    organ_like_target,
)


DEFAULT_GPU_SWEEP_SEEDS = tuple(range(5101, 5165))
DEFAULT_GPU_EVENT_MULTIPLIERS = (10, 20)
DEFAULT_RECORDS_PER_RUN = 20
SELECTED_GPU_TASK_IDS = (
    "s07_scrambled_ap_gradient",
    "s07_scrambled_boundary",
    "s07_scrambled_toy_organ_like",
    "s08_rotated_patch_boundary",
    "s09_scaled_scramble_ap_gradient_small",
    "s09_scaled_scramble_ap_gradient_large",
    "s10_symmetry_exact_vertical",
    "s10_symmetry_exact_horizontal",
)


@dataclass(frozen=True)
class GpuTaskTemplate:
    """One selected S07-S10 task family that can be tensorized."""

    task_id: str
    source_research_step_id: str
    source_task_type: str
    target: TargetMorphology
    initial_state_kind: str
    description: str
    scale_label: str = ""
    construction_axis: str = ""
    perturbation_type: str = ""

    @property
    def width(self) -> int:
        return int(self.target.substrate.dimensions.get("width", len(self.target.substrate.site_ids)))

    @property
    def height(self) -> int:
        return int(self.target.substrate.dimensions.get("height", 1))

    @property
    def site_count(self) -> int:
        return len(self.target.substrate.site_ids)


@dataclass(frozen=True)
class ScheduleTensors:
    """Precomputed actor and neighbor-order schedules shared by CPU/GPU runs."""

    actor_site_indices: torch.Tensor
    neighbor_order_indices: torch.Tensor
    schedule_seed: int


@dataclass(frozen=True)
class TensorTraceRecord:
    """Per-record tensor outputs for one batched simulation group."""

    event_step: int
    target_error: torch.Tensor
    displacement_fraction: torch.Tensor
    accepted_swaps: torch.Tensor
    attempted_swaps: torch.Tensor
    rejected_actions: torch.Tensor
    wait_actions: torch.Tensor
    energy_cost: torch.Tensor


@dataclass(frozen=True)
class TensorBatchOutput:
    """Tensor-kernel output for one task/policy/event-budget group."""

    final_occupants: torch.Tensor
    initial_target_error: torch.Tensor
    final_target_error: torch.Tensor
    initial_displacement_fraction: torch.Tensor
    final_displacement_fraction: torch.Tensor
    accepted_swaps: torch.Tensor
    attempted_swaps: torch.Tensor
    rejected_actions: torch.Tensor
    wait_actions: torch.Tensor
    total_energy_cost: torch.Tensor
    trace_records: tuple[TensorTraceRecord, ...]
    snapshots_by_step: Mapping[int, torch.Tensor]
    elapsed_seconds: float
    device_used: str


@dataclass(frozen=True)
class GpuSweepResult:
    """Complete S11 GPU sweep output before artifact serialization."""

    summary_rows: list[dict[str, Any]]
    trace_rows: list[dict[str, Any]]
    memory_rows: list[dict[str, Any]]
    vectorization_scope_rows: list[dict[str, Any]]
    snapshots_by_run_uid: dict[str, dict[int, list[int]]]
    templates_by_task_id: dict[str, GpuTaskTemplate]


def default_gpu_task_templates() -> tuple[GpuTaskTemplate, ...]:
    """Return selected S07-S10 square-grid, full-occupancy tasks for S11."""

    scaling_configs = {
        (config.source_target_id, config.scale_label): config
        for config in default_scaling_target_configs()
        if config.source_target_id == "ap_gradient"
    }
    symmetry_target = default_symmetry_target()
    return (
        GpuTaskTemplate(
            task_id="s07_scrambled_ap_gradient",
            source_research_step_id="S07",
            source_task_type="scrambled_embryo",
            target=gradient_target(6, 4),
            initial_state_kind="scrambled",
            description="S07-style fully scrambled anterior-posterior gradient target.",
        ),
        GpuTaskTemplate(
            task_id="s07_scrambled_boundary",
            source_research_step_id="S07",
            source_task_type="scrambled_embryo",
            target=boundary_target(6, 4),
            initial_state_kind="scrambled",
            description="S07-style fully scrambled boundary/interior target.",
        ),
        GpuTaskTemplate(
            task_id="s07_scrambled_toy_organ_like",
            source_research_step_id="S07",
            source_task_type="scrambled_embryo",
            target=organ_like_target(8, 5),
            initial_state_kind="scrambled",
            description="S07-style fully scrambled toy organ-like target.",
        ),
        GpuTaskTemplate(
            task_id="s08_rotated_patch_boundary",
            source_research_step_id="S08",
            source_task_type="rotated_patch_regeneration",
            target=boundary_target(6, 4),
            initial_state_kind="rotated_patch",
            perturbation_type="rotated_patch",
            description="S08 repairable rotated-patch perturbation on a boundary target.",
        ),
        GpuTaskTemplate(
            task_id="s09_scaled_scramble_ap_gradient_small",
            source_research_step_id="S09",
            source_task_type="scaled_scrambled_embryo",
            target=scaling_configs[("ap_gradient", "small")].make_target(),
            initial_state_kind="scrambled",
            scale_label="small",
            description="S09 no-retuning small-grid scaled scramble for the AP gradient.",
        ),
        GpuTaskTemplate(
            task_id="s09_scaled_scramble_ap_gradient_large",
            source_research_step_id="S09",
            source_task_type="scaled_scrambled_embryo",
            target=scaling_configs[("ap_gradient", "large")].make_target(),
            initial_state_kind="scrambled",
            scale_label="large",
            description="S09 no-retuning large-grid scaled scramble for the AP gradient.",
        ),
        GpuTaskTemplate(
            task_id="s10_symmetry_exact_vertical",
            source_research_step_id="S10",
            source_task_type="symmetry_breaking",
            target=symmetry_target,
            initial_state_kind="symmetry_exact_vertical",
            construction_axis="vertical",
            description="S10 exact vertical mirror-symmetric initial condition.",
        ),
        GpuTaskTemplate(
            task_id="s10_symmetry_exact_horizontal",
            source_research_step_id="S10",
            source_task_type="symmetry_breaking",
            target=symmetry_target,
            initial_state_kind="symmetry_exact_horizontal",
            construction_axis="horizontal",
            description="S10 exact horizontal mirror-symmetric initial condition.",
        ),
    )


def vectorization_scope_rows() -> tuple[dict[str, Any], ...]:
    """Document selected tensorized scope and explicit S11 blockers."""

    rows: list[dict[str, Any]] = []
    for template in default_gpu_task_templates():
        rows.append(
            {
                "research_step_id": "S11",
                "scope_id": template.task_id,
                "source_research_step_id": template.source_research_step_id,
                "source_task_type": template.source_task_type,
                "vectorized_in_s11": True,
                "status": "selected_square_grid_swap_wait_tensor_task",
                "reason": "Fully occupied square-grid task using S07 swap/wait policies and S04-compatible conservation accounting.",
            }
        )
    for perturbation_type in ("missing_patch", "frozen_patch", "duplicated_patch", "foreign_patch"):
        rows.append(
            {
                "research_step_id": "S11",
                "scope_id": f"s08_{perturbation_type}",
                "source_research_step_id": "S08",
                "source_task_type": perturbation_type,
                "vectorized_in_s11": False,
                "status": "blocked_semantics_not_tensorized",
                "reason": semantic_blocker_for_perturbation(perturbation_type),
            }
        )
    rows.append(
        {
            "research_step_id": "S11",
            "scope_id": "graph_or_irregular_substrates",
            "source_research_step_id": "S01-S10",
            "source_task_type": "non_square_grid_graph",
            "vectorized_in_s11": False,
            "status": "blocked_not_selected_for_tensor_kernel",
            "reason": "S11 kernel batches regular square-grid neighbor tensors only; graph substrates require separate ragged or sparse kernels.",
        }
    )
    return tuple(rows)


def initial_state_for_template(template: GpuTaskTemplate, seed: int) -> tuple[SubstrateState, dict[str, Any]]:
    """Construct one full-occupancy initial state and metadata for a template."""

    seed = int(seed)
    if template.initial_state_kind == "scrambled":
        state = scramble_target_state(template.target, seed)
        return state, {
            "perturbation_type": "full_scramble",
            "mask_id": "",
            "scale_label": template.scale_label,
            "construction_axis": template.construction_axis,
            "semantic_blocker": "",
        }
    if template.initial_state_kind == "rotated_patch":
        perturbed = apply_regeneration_perturbation(template.target, "rotated_patch", seed)
        return perturbed.state, {
            "perturbation_type": "rotated_patch",
            "mask_id": perturbed.mask.mask_id,
            "mask_area": len(perturbed.mask.site_ids),
            "mask_width": perturbed.mask.width,
            "mask_height": perturbed.mask.height,
            "scale_label": template.scale_label,
            "construction_axis": template.construction_axis,
            "semantic_blocker": perturbed.semantic_blocker,
        }
    if template.initial_state_kind.startswith("symmetry_exact_"):
        axis = template.initial_state_kind.replace("symmetry_exact_", "")
        condition = next(
            item
            for item in default_initial_conditions()
            if item.construction_axis == axis and item.symmetry_class == "exact_axis_mirror"
        )
        state = construct_axis_symmetric_initial_state(template.target, condition, seed)
        return state, {
            "perturbation_type": "axis_symmetric_initial_state",
            "mask_id": "",
            "scale_label": template.scale_label,
            "construction_axis": axis,
            "semantic_blocker": "",
        }
    raise ValueError(f"unsupported S11 initial_state_kind: {template.initial_state_kind}")


def target_distance_matrix(target: TargetMorphology, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return matrix[cell_index, target_site_index] of S03 identity distances."""

    _require_dense_site_ids(target)
    constructed = target.constructed_substrate()
    cells = [constructed.cell_at(site_id) for site_id in target.substrate.site_ids]
    rows = []
    for cell in cells:
        if cell is None:
            raise ValueError("constructed target unexpectedly has an empty site")
        identity = identity_from_substrate_cell(cell, fallback_scalar=False)
        rows.append(
            [
                float(target.schema.distance(identity, target.identities_by_site[int(site_id)]))
                for site_id in target.substrate.site_ids
            ]
        )
    return torch.tensor(rows, dtype=dtype)


def neighbor_index_tensor(target: TargetMorphology) -> torch.Tensor:
    """Return padded neighbor indices for a dense square-grid target."""

    _require_dense_site_ids(target)
    site_ids = tuple(target.substrate.site_ids)
    max_degree = max(len(target.substrate.neighbors(site_id)) for site_id in site_ids)
    rows = []
    for site_id in site_ids:
        neighbors = list(target.substrate.neighbors(site_id))
        rows.append(neighbors + [-1] * (max_degree - len(neighbors)))
    return torch.tensor(rows, dtype=torch.long)


def state_to_occupant_indices(target: TargetMorphology, state: SubstrateState) -> tuple[int, ...]:
    """Encode a full-occupancy state as target-constructed cell indices."""

    _require_dense_site_ids(target)
    constructed = target.constructed_substrate()
    cell_to_index = {
        str(constructed.cell_at(site_id).cell_id): int(site_id)
        for site_id in target.substrate.site_ids
        if constructed.cell_at(site_id) is not None
    }
    occupants = []
    for site_id in target.substrate.site_ids:
        cell = state.cell_at(site_id)
        if cell is None:
            raise ValueError("S11 tensorized tasks require full occupancy")
        cell_id = str(cell.cell_id)
        if cell_id not in cell_to_index:
            raise ValueError(f"state contains cell outside target population: {cell_id}")
        occupants.append(cell_to_index[cell_id])
    return tuple(occupants)


def occupant_indices_to_state(target: TargetMorphology, occupant_indices: Sequence[int]) -> SubstrateState:
    """Decode occupant indices into a SubstrateState for reference checks."""

    _require_dense_site_ids(target)
    if len(occupant_indices) != len(target.substrate.site_ids):
        raise ValueError("occupant_indices length does not match target")
    constructed = target.constructed_substrate()
    cells = [constructed.cell_at(site_id) for site_id in target.substrate.site_ids]
    state = target.substrate.copy_empty()
    for site_id, occupant_idx in zip(target.substrate.site_ids, occupant_indices, strict=True):
        state.place_cell(site_id, cells[int(occupant_idx)])
    return state


def make_deterministic_schedules(
    *,
    batch_size: int,
    event_cap: int,
    site_count: int,
    max_degree: int,
    schedule_seed: int,
) -> ScheduleTensors:
    """Precompute deterministic actor choices and shuffled neighbor orders."""

    if batch_size <= 0 or event_cap <= 0 or site_count <= 0 or max_degree <= 0:
        raise ValueError("batch_size, event_cap, site_count, and max_degree must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(schedule_seed) % (2**63 - 1))
    actor_site_indices = torch.randint(0, int(site_count), (int(batch_size), int(event_cap)), generator=generator)
    neighbor_scores = torch.rand((int(batch_size), int(event_cap), int(max_degree)), generator=generator)
    neighbor_order_indices = torch.argsort(neighbor_scores, dim=-1)
    return ScheduleTensors(
        actor_site_indices=actor_site_indices.to(dtype=torch.long),
        neighbor_order_indices=neighbor_order_indices.to(dtype=torch.long),
        schedule_seed=int(schedule_seed),
    )


def schedule_seed_for_group(
    *,
    template: GpuTaskTemplate,
    policy_id: str,
    event_multiplier: int,
    seeds: Sequence[int],
) -> int:
    """Return a stable CPU-generated schedule seed for a batched group."""

    seed_text = f"S11:{template.task_id}:{policy_id}:{int(event_multiplier)}:{','.join(str(seed) for seed in seeds)}"
    return 7_000_000 + stable_int_hash(seed_text)


def run_tensor_batch(
    *,
    initial_occupants: torch.Tensor,
    distance_matrix: torch.Tensor,
    neighbor_indices: torch.Tensor,
    policy_id: str,
    schedules: ScheduleTensors,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
    device: str | torch.device = "cpu",
    snapshot_steps: Sequence[int] | None = None,
) -> TensorBatchOutput:
    """Run one tensorized S07 local-control batch on CPU or GPU."""

    if policy_id not in RUNNABLE_POLICY_IDS:
        raise ValueError(f"unsupported S11 policy_id: {policy_id}")
    device_obj = torch.device(device)
    started = time.perf_counter()
    occupants = initial_occupants.to(device_obj, dtype=torch.long).clone()
    if occupants.ndim != 2:
        raise ValueError("initial_occupants must be a [batch, sites] tensor")
    if int(occupants.min().item()) < 0:
        raise ValueError("S11 tensorized tasks require full occupancy")
    batch_size, site_count = occupants.shape
    event_cap = int(schedules.actor_site_indices.shape[1])
    if schedules.actor_site_indices.shape != (batch_size, event_cap):
        raise ValueError("actor schedule shape does not match initial_occupants")
    max_degree = int(neighbor_indices.shape[1])
    if schedules.neighbor_order_indices.shape != (batch_size, event_cap, max_degree):
        raise ValueError("neighbor order schedule shape does not match neighbor tensor")

    distance = distance_matrix.to(device_obj, dtype=torch.float64)
    neighbors = neighbor_indices.to(device_obj, dtype=torch.long)
    actor_schedule = schedules.actor_site_indices.to(device_obj, dtype=torch.long)
    order_schedule = schedules.neighbor_order_indices.to(device_obj, dtype=torch.long)
    run_indices = torch.arange(batch_size, device=device_obj)
    record_every = max(1, event_cap // max(1, int(records_per_run)))
    snapshot_set = set(int(step) for step in (snapshot_steps or (0, event_cap // 2, event_cap)))
    snapshot_set.add(0)
    snapshot_set.add(event_cap)

    accepted_swaps = torch.zeros(batch_size, dtype=torch.long, device=device_obj)
    attempted_swaps = torch.zeros(batch_size, dtype=torch.long, device=device_obj)
    rejected_actions = torch.zeros(batch_size, dtype=torch.long, device=device_obj)
    wait_actions = torch.zeros(batch_size, dtype=torch.long, device=device_obj)
    energy_cost = torch.zeros(batch_size, dtype=torch.float64, device=device_obj)

    initial_error = _target_error_vector(occupants, distance)
    initial_displacement = _displacement_vector(occupants, distance)
    trace_records = [
        _tensor_trace_record(
            0,
            occupants,
            distance,
            accepted_swaps,
            attempted_swaps,
            rejected_actions,
            wait_actions,
            energy_cost,
        )
    ]
    snapshots: dict[int, torch.Tensor] = {}
    if 0 in snapshot_set:
        snapshots[0] = occupants.detach().cpu().clone()

    for event_idx in range(event_cap):
        actor = actor_schedule[:, event_idx]
        order = order_schedule[:, event_idx, :]
        ordered_neighbors = torch.gather(neighbors[actor], 1, order)
        if policy_id == "s07_random_adjacent_swap_control":
            chosen_site = _first_valid_neighbor(ordered_neighbors)
            propose_swap = chosen_site >= 0
        else:
            chosen_site, propose_swap = _local_target_descent_choice(
                occupants=occupants,
                distance=distance,
                actor=actor,
                ordered_neighbors=ordered_neighbors,
                run_indices=run_indices,
            )

        valid_actor = occupants[run_indices, actor] >= 0
        chosen_clamped = chosen_site.clamp_min(0)
        valid_target = (chosen_site >= 0) & (occupants[run_indices, chosen_clamped] >= 0)
        requested_swap = propose_swap & valid_actor
        allowed_swap = requested_swap & valid_target
        rejected_actions += (requested_swap & ~allowed_swap).to(torch.long)
        wait_actions += (~requested_swap).to(torch.long)
        attempted_swaps += requested_swap.to(torch.long)
        if bool(allowed_swap.any().item()):
            mask = allowed_swap
            src = actor[mask]
            tgt = chosen_site[mask]
            masked_runs = run_indices[mask]
            src_values = occupants[masked_runs, src].clone()
            tgt_values = occupants[masked_runs, tgt].clone()
            occupants[masked_runs, src] = tgt_values
            occupants[masked_runs, tgt] = src_values
            accepted_swaps += allowed_swap.to(torch.long)
            energy_cost += allowed_swap.to(torch.float64)

        event_step = event_idx + 1
        if event_step in snapshot_set:
            snapshots[event_step] = occupants.detach().cpu().clone()
        if event_step % record_every == 0 or event_step == event_cap:
            trace_records.append(
                _tensor_trace_record(
                    event_step,
                    occupants,
                    distance,
                    accepted_swaps,
                    attempted_swaps,
                    rejected_actions,
                    wait_actions,
                    energy_cost,
                )
            )

    final_error = _target_error_vector(occupants, distance)
    final_displacement = _displacement_vector(occupants, distance)
    if event_cap not in snapshots:
        snapshots[event_cap] = occupants.detach().cpu().clone()
    if torch.device(device_obj).type == "cuda":
        torch.cuda.synchronize(device_obj)
    elapsed = time.perf_counter() - started
    return TensorBatchOutput(
        final_occupants=occupants.detach().cpu(),
        initial_target_error=initial_error.detach().cpu(),
        final_target_error=final_error.detach().cpu(),
        initial_displacement_fraction=initial_displacement.detach().cpu(),
        final_displacement_fraction=final_displacement.detach().cpu(),
        accepted_swaps=accepted_swaps.detach().cpu(),
        attempted_swaps=attempted_swaps.detach().cpu(),
        rejected_actions=rejected_actions.detach().cpu(),
        wait_actions=wait_actions.detach().cpu(),
        total_energy_cost=energy_cost.detach().cpu(),
        trace_records=tuple(trace_records),
        snapshots_by_step=snapshots,
        elapsed_seconds=float(elapsed),
        device_used=str(device_obj),
    )


def run_gpu_tissue_sweep(
    *,
    templates: Sequence[GpuTaskTemplate] | None = None,
    policy_ids: Sequence[str] = RUNNABLE_POLICY_IDS,
    seeds: Sequence[int] = DEFAULT_GPU_SWEEP_SEEDS,
    event_multipliers: Sequence[int] = DEFAULT_GPU_EVENT_MULTIPLIERS,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
    device: str | torch.device = "cuda",
) -> GpuSweepResult:
    """Run the selected S11 GPU-batched task sweep."""

    templates = tuple(templates or default_gpu_task_templates())
    seeds = tuple(int(seed) for seed in seeds)
    event_multipliers = tuple(int(value) for value in event_multipliers)
    if not seeds:
        raise ValueError("at least one seed is required")
    if not event_multipliers:
        raise ValueError("at least one event multiplier is required")
    for policy_id in policy_ids:
        if policy_id not in RUNNABLE_POLICY_IDS:
            raise ValueError(f"unsupported policy_id: {policy_id}")

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for S11 but torch.cuda.is_available() is false")
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    snapshots_by_run_uid: dict[str, dict[int, list[int]]] = {}
    templates_by_task_id = {template.task_id: template for template in templates}

    for template in templates:
        distance = target_distance_matrix(template.target)
        neighbors = neighbor_index_tensor(template.target)
        initial_states = []
        initial_metadata = []
        for seed in seeds:
            state, metadata = initial_state_for_template(template, seed)
            initial_states.append(state)
            initial_metadata.append(metadata)
        initial_occupants = torch.tensor(
            [state_to_occupant_indices(template.target, state) for state in initial_states],
            dtype=torch.long,
        )
        for policy_id in policy_ids:
            policy = policy_by_id(policy_id)
            for event_multiplier in event_multipliers:
                event_cap = int(event_multiplier) * template.site_count
                schedule_seed = schedule_seed_for_group(
                    template=template,
                    policy_id=policy_id,
                    event_multiplier=event_multiplier,
                    seeds=seeds,
                )
                schedules = make_deterministic_schedules(
                    batch_size=len(seeds),
                    event_cap=event_cap,
                    site_count=template.site_count,
                    max_degree=int(neighbors.shape[1]),
                    schedule_seed=schedule_seed,
                )
                memory_before = cuda_memory_snapshot(device_obj, "before")
                if device_obj.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device_obj)
                output = run_tensor_batch(
                    initial_occupants=initial_occupants,
                    distance_matrix=distance,
                    neighbor_indices=neighbors,
                    policy_id=policy_id,
                    schedules=schedules,
                    records_per_run=records_per_run,
                    device=device_obj,
                )
                memory_after = cuda_memory_snapshot(device_obj, "after")
                memory_peak = cuda_peak_memory_snapshot(device_obj)
                group_event_count = int(len(seeds) * event_cap)
                events_per_second = float(group_event_count / max(output.elapsed_seconds, 1e-12))
                memory_rows.append(
                    {
                        "research_step_id": "S11",
                        "task_id": template.task_id,
                        "source_research_step_id": template.source_research_step_id,
                        "policy_id": policy_id,
                        "event_multiplier": int(event_multiplier),
                        "event_cap": int(event_cap),
                        "batch_size": int(len(seeds)),
                        "device_requested": str(device_obj),
                        "device_used": output.device_used,
                        "cuda_available": bool(torch.cuda.is_available()),
                        "cuda_device_name": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "",
                        "cuda_total_memory_bytes": memory_after.get("total_memory_bytes", 0),
                        "allocated_before_bytes": memory_before.get("memory_allocated_bytes", 0),
                        "reserved_before_bytes": memory_before.get("memory_reserved_bytes", 0),
                        "allocated_after_bytes": memory_after.get("memory_allocated_bytes", 0),
                        "reserved_after_bytes": memory_after.get("memory_reserved_bytes", 0),
                        "peak_allocated_bytes": memory_peak.get("max_memory_allocated_bytes", 0),
                        "peak_reserved_bytes": memory_peak.get("max_memory_reserved_bytes", 0),
                        "elapsed_seconds": output.elapsed_seconds,
                        "events_per_second": events_per_second,
                    }
                )
                for run_index, seed in enumerate(seeds):
                    run_uid = f"{template.task_id}|{policy_id}|m{event_multiplier}|seed{seed}"
                    row = _summary_row(
                        template=template,
                        policy=policy,
                        seed=seed,
                        event_multiplier=event_multiplier,
                        event_cap=event_cap,
                        records_per_run=records_per_run,
                        schedule_seed=schedule_seed,
                        batch_size=len(seeds),
                        run_index=run_index,
                        metadata=initial_metadata[run_index],
                        output=output,
                        events_per_second=events_per_second,
                        run_uid=run_uid,
                    )
                    summary_rows.append(row)
                    snapshots_by_run_uid[run_uid] = {
                        int(step): output.snapshots_by_step[int(step)][run_index].to(dtype=torch.long).tolist()
                        for step in sorted(output.snapshots_by_step)
                    }
                for record in output.trace_records:
                    for run_index, seed in enumerate(seeds):
                        run_uid = f"{template.task_id}|{policy_id}|m{event_multiplier}|seed{seed}"
                        trace_rows.append(
                            {
                                "research_step_id": "S11",
                                "run_uid": run_uid,
                                "task_id": template.task_id,
                                "source_research_step_id": template.source_research_step_id,
                                "source_task_type": template.source_task_type,
                                "target_id": template.target.target_id,
                                "target_kind": template.target.target_kind,
                                "policy_id": policy_id,
                                "simulation_seed": int(seed),
                                "event_multiplier": int(event_multiplier),
                                "event_cap": int(event_cap),
                                "event_step": int(record.event_step),
                                "target_error": float(record.target_error[run_index].item()),
                                "displacement_fraction": float(record.displacement_fraction[run_index].item()),
                                "accepted_swaps": int(record.accepted_swaps[run_index].item()),
                                "attempted_swaps": int(record.attempted_swaps[run_index].item()),
                                "rejected_actions": int(record.rejected_actions[run_index].item()),
                                "wait_actions": int(record.wait_actions[run_index].item()),
                                "energy_cost": float(record.energy_cost[run_index].item()),
                            }
                        )

    return GpuSweepResult(
        summary_rows=summary_rows,
        trace_rows=trace_rows,
        memory_rows=memory_rows,
        vectorization_scope_rows=list(vectorization_scope_rows()),
        snapshots_by_run_uid=snapshots_by_run_uid,
        templates_by_task_id=templates_by_task_id,
    )


def validate_cpu_gpu_agreement(
    *,
    templates: Sequence[GpuTaskTemplate] | None = None,
    policy_ids: Sequence[str] = RUNNABLE_POLICY_IDS,
    seeds: Sequence[int] = (6101, 6102, 6103),
    event_multiplier: int = 4,
) -> list[dict[str, Any]]:
    """Compare CPU and GPU tensor kernels on small deterministic fixtures."""

    selected = tuple(templates or default_gpu_task_templates())[:3]
    rows: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return [
            {
                "research_step_id": "S11",
                "validation_case": "cpu_gpu_tensor_agreement",
                "success": False,
                "detail": "CUDA is not available, so CPU-vs-GPU agreement could not be tested.",
                "max_target_error_abs_diff": math.nan,
                "final_occupants_equal": False,
                "counts_equal": False,
            }
        ]
    for template in selected:
        distance = target_distance_matrix(template.target)
        neighbors = neighbor_index_tensor(template.target)
        initial_occupants = torch.tensor(
            [
                state_to_occupant_indices(template.target, initial_state_for_template(template, seed)[0])
                for seed in seeds
            ],
            dtype=torch.long,
        )
        for policy_id in policy_ids:
            event_cap = int(event_multiplier) * template.site_count
            schedules = make_deterministic_schedules(
                batch_size=len(seeds),
                event_cap=event_cap,
                site_count=template.site_count,
                max_degree=int(neighbors.shape[1]),
                schedule_seed=schedule_seed_for_group(
                    template=template,
                    policy_id=policy_id,
                    event_multiplier=event_multiplier,
                    seeds=seeds,
                ),
            )
            cpu = run_tensor_batch(
                initial_occupants=initial_occupants,
                distance_matrix=distance,
                neighbor_indices=neighbors,
                policy_id=policy_id,
                schedules=schedules,
                records_per_run=4,
                device="cpu",
            )
            gpu = run_tensor_batch(
                initial_occupants=initial_occupants,
                distance_matrix=distance,
                neighbor_indices=neighbors,
                policy_id=policy_id,
                schedules=schedules,
                records_per_run=4,
                device="cuda",
            )
            error_diff = torch.max(torch.abs(cpu.final_target_error - gpu.final_target_error)).item()
            final_equal = bool(torch.equal(cpu.final_occupants, gpu.final_occupants))
            counts_equal = bool(
                torch.equal(cpu.accepted_swaps, gpu.accepted_swaps)
                and torch.equal(cpu.attempted_swaps, gpu.attempted_swaps)
                and torch.equal(cpu.wait_actions, gpu.wait_actions)
                and torch.equal(cpu.rejected_actions, gpu.rejected_actions)
            )
            rows.append(
                {
                    "research_step_id": "S11",
                    "validation_case": "cpu_gpu_tensor_agreement",
                    "task_id": template.task_id,
                    "policy_id": policy_id,
                    "seed_count": len(seeds),
                    "event_multiplier": int(event_multiplier),
                    "success": bool(final_equal and counts_equal and error_diff <= 1e-12),
                    "max_target_error_abs_diff": float(error_diff),
                    "final_occupants_equal": final_equal,
                    "counts_equal": counts_equal,
                    "detail": "CPU and CUDA tensor kernels used identical precomputed schedules.",
                }
            )
    return rows


def validate_reference_cpu_agreement(
    *,
    templates: Sequence[GpuTaskTemplate] | None = None,
    policy_ids: Sequence[str] = RUNNABLE_POLICY_IDS,
    seeds: Sequence[int] = (6201, 6202),
    event_multiplier: int = 3,
) -> list[dict[str, Any]]:
    """Compare tensor CPU output to object-state S04-compatible replay."""

    selected = tuple(templates or default_gpu_task_templates())[:4]
    rows: list[dict[str, Any]] = []
    for template in selected:
        distance = target_distance_matrix(template.target)
        neighbors = neighbor_index_tensor(template.target)
        initial_states = [initial_state_for_template(template, seed)[0] for seed in seeds]
        initial_occupants = torch.tensor(
            [state_to_occupant_indices(template.target, state) for state in initial_states],
            dtype=torch.long,
        )
        for policy_id in policy_ids:
            event_cap = int(event_multiplier) * template.site_count
            schedules = make_deterministic_schedules(
                batch_size=len(seeds),
                event_cap=event_cap,
                site_count=template.site_count,
                max_degree=int(neighbors.shape[1]),
                schedule_seed=schedule_seed_for_group(
                    template=template,
                    policy_id=policy_id,
                    event_multiplier=event_multiplier,
                    seeds=seeds,
                ),
            )
            tensor_output = run_tensor_batch(
                initial_occupants=initial_occupants,
                distance_matrix=distance,
                neighbor_indices=neighbors,
                policy_id=policy_id,
                schedules=schedules,
                records_per_run=3,
                device="cpu",
            )
            final_equal = True
            counts_equal = True
            max_error_diff = 0.0
            for run_index, state in enumerate(initial_states):
                replay = replay_reference_cpu_from_schedule(
                    target=template.target,
                    initial_state=state,
                    policy_id=policy_id,
                    actor_schedule=schedules.actor_site_indices[run_index].tolist(),
                    neighbor_order_schedule=schedules.neighbor_order_indices[run_index].tolist(),
                )
                final_indices = state_to_occupant_indices(template.target, replay["final_state"])
                final_equal = final_equal and tuple(final_indices) == tuple(
                    int(value) for value in tensor_output.final_occupants[run_index].tolist()
                )
                counts_equal = counts_equal and int(tensor_output.accepted_swaps[run_index].item()) == replay["accepted_swaps"]
                counts_equal = counts_equal and int(tensor_output.attempted_swaps[run_index].item()) == replay["attempted_swaps"]
                counts_equal = counts_equal and int(tensor_output.wait_actions[run_index].item()) == replay["wait_actions"]
                counts_equal = counts_equal and int(tensor_output.rejected_actions[run_index].item()) == replay["rejected_actions"]
                max_error_diff = max(
                    max_error_diff,
                    abs(float(tensor_output.final_target_error[run_index].item()) - float(template.target.target_error(replay["final_state"]))),
                )
            rows.append(
                {
                    "research_step_id": "S11",
                    "validation_case": "tensor_cpu_reference_replay_agreement",
                    "task_id": template.task_id,
                    "policy_id": policy_id,
                    "seed_count": len(seeds),
                    "event_multiplier": int(event_multiplier),
                    "success": bool(final_equal and counts_equal and max_error_diff <= 1e-12),
                    "max_target_error_abs_diff": float(max_error_diff),
                    "final_occupants_equal": bool(final_equal),
                    "counts_equal": bool(counts_equal),
                    "detail": "Tensor CPU kernel matched object-state replay under the same actor and neighbor-order schedule.",
                }
            )
    return rows


def replay_reference_cpu_from_schedule(
    *,
    target: TargetMorphology,
    initial_state: SubstrateState,
    policy_id: str,
    actor_schedule: Sequence[int],
    neighbor_order_schedule: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Replay one scheduled run through SubstrateState and S09 fast actions."""

    if len(actor_schedule) != len(neighbor_order_schedule):
        raise ValueError("actor and neighbor schedules must have the same length")
    state = clone_substrate_state(initial_state)
    distance_lookup = target_identity_distance_lookup(target)
    accepted_swaps = 0
    attempted_swaps = 0
    rejected_actions = 0
    wait_actions = 0
    energy_cost = 0.0
    population = len(target.substrate.site_ids)
    for actor_site_id, order in zip(actor_schedule, neighbor_order_schedule, strict=True):
        actor_site_id = int(actor_site_id)
        neighbors = list(target.substrate.neighbors(actor_site_id))
        ordered_neighbors = [neighbors[int(idx)] for idx in order if int(idx) < len(neighbors)]
        target_site_id: int | None = None
        if state.cell_at(actor_site_id) is None:
            request = MorphogenesisActionRequest("wait", actor_site_id)
        elif policy_id == "s07_random_adjacent_swap_control":
            target_site_id = ordered_neighbors[0] if ordered_neighbors else None
            request = (
                MorphogenesisActionRequest("swap", actor_site_id, target_site_id)
                if target_site_id is not None
                else MorphogenesisActionRequest("wait", actor_site_id)
            )
        elif policy_id == "s07_local_target_neighbor_descent":
            best_delta = 0.0
            for neighbor_site_id in ordered_neighbors:
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
                    target_site_id = int(neighbor_site_id)
            request = (
                MorphogenesisActionRequest("swap", actor_site_id, target_site_id)
                if target_site_id is not None
                else MorphogenesisActionRequest("wait", actor_site_id)
            )
        else:
            raise ValueError(f"unsupported S11 policy_id: {policy_id}")

        result = execute_swap_wait_fast(state, request, population_current=population)
        if request.action_type == "swap":
            attempted_swaps += 1
        if request.action_type == "wait":
            wait_actions += 1
        if not result.allowed:
            rejected_actions += 1
        if result.allowed and result.action_type == "swap" and result.state_changed:
            accepted_swaps += 1
        energy_cost += float(result.energy_cost_charged)
    return {
        "final_state": state,
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "wait_actions": int(wait_actions),
        "energy_cost": float(energy_cost),
    }


def cuda_memory_snapshot(device: str | torch.device, label: str) -> dict[str, Any]:
    """Return PyTorch CUDA memory counters, or zero counters for CPU."""

    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return {
            "label": label,
            "memory_allocated_bytes": 0,
            "memory_reserved_bytes": 0,
            "free_memory_bytes": 0,
            "total_memory_bytes": 0,
        }
    free_memory, total_memory = torch.cuda.mem_get_info(device_obj)
    return {
        "label": label,
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device_obj)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device_obj)),
        "free_memory_bytes": int(free_memory),
        "total_memory_bytes": int(total_memory),
    }


def cuda_peak_memory_snapshot(device: str | torch.device) -> dict[str, int]:
    """Return PyTorch CUDA peak memory counters, or zeros for CPU."""

    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return {"max_memory_allocated_bytes": 0, "max_memory_reserved_bytes": 0}
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)),
    }


def _summary_row(
    *,
    template: GpuTaskTemplate,
    policy: Any,
    seed: int,
    event_multiplier: int,
    event_cap: int,
    records_per_run: int,
    schedule_seed: int,
    batch_size: int,
    run_index: int,
    metadata: Mapping[str, Any],
    output: TensorBatchOutput,
    events_per_second: float,
    run_uid: str,
) -> dict[str, Any]:
    initial_error = float(output.initial_target_error[run_index].item())
    final_error = float(output.final_target_error[run_index].item())
    accepted_swaps = int(output.accepted_swaps[run_index].item())
    attempted_swaps = int(output.attempted_swaps[run_index].item())
    rejected_actions = int(output.rejected_actions[run_index].item())
    wait_actions = int(output.wait_actions[run_index].item())
    energy_cost = float(output.total_energy_cost[run_index].item())
    return {
        "research_step_id": "S11",
        "run_uid": run_uid,
        "source_research_step_id": template.source_research_step_id,
        "source_task_type": template.source_task_type,
        "task_id": template.task_id,
        "target_id": template.target.target_id,
        "target_kind": template.target.target_kind,
        "target_hash": template.target.target_hash,
        "substrate_kind": template.target.substrate.kind,
        "width": template.width,
        "height": template.height,
        "site_count": template.site_count,
        "policy_id": policy.policy_id,
        "policy_family": policy.policy_family,
        "declared_information_access": policy.declared_information_access,
        "simulation_seed": int(seed),
        "event_multiplier": int(event_multiplier),
        "event_cap": int(event_cap),
        "records_per_run": int(records_per_run),
        "schedule_seed": int(schedule_seed),
        "batch_size": int(batch_size),
        "tensor_backend": "torch",
        "device_used": output.device_used,
        "vectorized": True,
        "initial_target_error": initial_error,
        "final_target_error": final_error,
        "delta_target_error": final_error - initial_error,
        "target_recovery_fraction": recovery_fraction(initial_error, final_error),
        "initial_displacement_fraction": float(output.initial_displacement_fraction[run_index].item()),
        "final_displacement_fraction": float(output.final_displacement_fraction[run_index].item()),
        "accepted_swaps": accepted_swaps,
        "attempted_swaps": attempted_swaps,
        "rejected_actions": rejected_actions,
        "wait_actions": wait_actions,
        "total_energy_cost": energy_cost,
        "energy_per_site": float(energy_cost / max(1, template.site_count)),
        "accepted_swaps_per_site": float(accepted_swaps / max(1, template.site_count)),
        "attempted_swaps_per_site": float(attempted_swaps / max(1, template.site_count)),
        "population_initial": template.site_count,
        "population_final": template.site_count,
        "population_min": template.site_count,
        "population_max": template.site_count,
        "population_delta_total": 0,
        "group_elapsed_seconds": output.elapsed_seconds,
        "group_events_per_second": float(events_per_second),
        "initial_state_kind": template.initial_state_kind,
        "description": template.description,
        "scale_label": str(metadata.get("scale_label", "")),
        "perturbation_type": str(metadata.get("perturbation_type", "")),
        "mask_id": str(metadata.get("mask_id", "")),
        "mask_area": int(metadata.get("mask_area", 0)),
        "mask_width": int(metadata.get("mask_width", 0)),
        "mask_height": int(metadata.get("mask_height", 0)),
        "construction_axis": str(metadata.get("construction_axis", "")),
        "semantic_blocker": str(metadata.get("semantic_blocker", "")),
    }


def _target_error_vector(occupants: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    site_indices = torch.arange(occupants.shape[1], device=occupants.device).unsqueeze(0).expand_as(occupants)
    return distance[occupants, site_indices].mean(dim=1)


def _displacement_vector(occupants: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    site_indices = torch.arange(occupants.shape[1], device=occupants.device).unsqueeze(0).expand_as(occupants)
    return (distance[occupants, site_indices] > 1e-12).to(torch.float64).mean(dim=1)


def _tensor_trace_record(
    event_step: int,
    occupants: torch.Tensor,
    distance: torch.Tensor,
    accepted_swaps: torch.Tensor,
    attempted_swaps: torch.Tensor,
    rejected_actions: torch.Tensor,
    wait_actions: torch.Tensor,
    energy_cost: torch.Tensor,
) -> TensorTraceRecord:
    return TensorTraceRecord(
        event_step=int(event_step),
        target_error=_target_error_vector(occupants, distance).detach().cpu(),
        displacement_fraction=_displacement_vector(occupants, distance).detach().cpu(),
        accepted_swaps=accepted_swaps.detach().cpu().clone(),
        attempted_swaps=attempted_swaps.detach().cpu().clone(),
        rejected_actions=rejected_actions.detach().cpu().clone(),
        wait_actions=wait_actions.detach().cpu().clone(),
        energy_cost=energy_cost.detach().cpu().clone(),
    )


def _first_valid_neighbor(ordered_neighbors: torch.Tensor) -> torch.Tensor:
    chosen = torch.full(
        (ordered_neighbors.shape[0],),
        -1,
        dtype=torch.long,
        device=ordered_neighbors.device,
    )
    for neighbor_slot in range(ordered_neighbors.shape[1]):
        candidate = ordered_neighbors[:, neighbor_slot]
        take = (chosen < 0) & (candidate >= 0)
        chosen = torch.where(take, candidate, chosen)
    return chosen


def _local_target_descent_choice(
    *,
    occupants: torch.Tensor,
    distance: torch.Tensor,
    actor: torch.Tensor,
    ordered_neighbors: torch.Tensor,
    run_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = torch.full((occupants.shape[0],), -1, dtype=torch.long, device=occupants.device)
    best_delta = torch.zeros(occupants.shape[0], dtype=torch.float64, device=occupants.device)
    actor_cell = occupants[run_indices, actor]
    actor_current_error = distance[actor_cell, actor]
    for neighbor_slot in range(ordered_neighbors.shape[1]):
        candidate = ordered_neighbors[:, neighbor_slot]
        valid = candidate >= 0
        candidate_clamped = candidate.clamp_min(0)
        neighbor_cell = occupants[run_indices, candidate_clamped]
        delta = (
            distance[neighbor_cell, actor]
            + distance[actor_cell, candidate_clamped]
            - actor_current_error
            - distance[neighbor_cell, candidate_clamped]
        )
        better = valid & (delta < best_delta - 1e-12)
        chosen = torch.where(better, candidate, chosen)
        best_delta = torch.where(better, delta, best_delta)
    return chosen, chosen >= 0


def _require_dense_site_ids(target: TargetMorphology) -> None:
    if target.substrate.kind != "square_grid_2d":
        raise ValueError(f"S11 tensor kernel only supports square_grid_2d targets, not {target.substrate.kind}")
    expected = tuple(range(len(target.substrate.site_ids)))
    if tuple(target.substrate.site_ids) != expected:
        raise ValueError("S11 tensor kernel requires dense 0..N-1 site IDs")
