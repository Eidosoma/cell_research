"""Scrambled-embryo benchmark helpers for E05 S07.

The benchmark intentionally separates two concepts:

* available E03/E04 policy artifacts are inventoried and mapped only when their
  declared observation contracts can support the S03 2D target task; and
* runnable E05 controls use S01-S05 components to provide bounded recovery
  trajectories without claiming to be transferred E03/E04 policies.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from src.e05.actions import ActionExecutor, MorphogenesisActionRequest
from src.e05.cell_identity import identity_from_substrate_cell
from src.e05.morphospace_metrics import aggregate_morphospace_error, evaluate_morphology_metrics, metric_result_rows
from src.e05.substrates import SubstrateState
from src.e05.targets import TargetMorphology, default_target_gallery


RUNNABLE_POLICY_IDS = ("s07_local_target_neighbor_descent", "s07_random_adjacent_swap_control")
DEFAULT_2D_TARGET_KINDS = ("gradient", "stripes", "boundary", "ring", "symmetry", "organ_like")
DEFAULT_SEEDS = (1101, 1102, 1103, 1104, 1105)
DEFAULT_EVENT_MULTIPLIER = 80
DEFAULT_RECORDS_PER_RUN = 40
POLICY_SOURCE_PATHS = (
    Path("/previous-artifacts/E03/policies/e03_classic_policy_library.json"),
    Path("/previous-artifacts/E03/policies/e03_frontier_candidate_policies.jsonl"),
    Path("/previous-artifacts/E03/policies/e03_generated_policy_library.jsonl"),
    Path("/previous-artifacts/E03/policies/e03_qd_discovered_policies.jsonl"),
    Path("/previous-artifacts/E04/policies/e04_evolved_repair_policies.jsonl"),
    Path("/previous-artifacts/E04/reports/e04_local_learning_rule_spec.md"),
)


class ScrambledPolicy(Protocol):
    policy_id: str
    policy_family: str
    declared_information_access: str

    def propose(
        self,
        *,
        state: SubstrateState,
        target: TargetMorphology,
        actor_site_id: int,
        rng: random.Random,
    ) -> MorphogenesisActionRequest:
        """Return a local S04 action request for one actor site."""


@dataclass(frozen=True)
class ScrambledRunResult:
    """One S07 benchmark trajectory and summary."""

    summary_row: dict[str, Any]
    trace_rows: tuple[dict[str, Any], ...]
    metric_rows: tuple[dict[str, Any], ...]
    initial_state: SubstrateState
    final_state: SubstrateState


class LocalTargetNeighborDescentPolicy:
    """Local target-aware adjacent-swap descent control.

    The policy only evaluates the actor site and its immediate neighbors.  It
    does see the target identities at those same local sites, so it is a bounded
    E05 control rather than an E03/E04 transferred policy.
    """

    policy_id = "s07_local_target_neighbor_descent"
    policy_family = "e05_local_target_aware_control"
    declared_information_access = (
        "actor identity, adjacent-neighbor identities, and target identities at the actor and adjacent-neighbor sites"
    )

    def propose(
        self,
        *,
        state: SubstrateState,
        target: TargetMorphology,
        actor_site_id: int,
        rng: random.Random,
    ) -> MorphogenesisActionRequest:
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
            delta = adjacent_swap_target_error_delta(target, state, actor_site_id, neighbor_site_id)
            if delta < best_delta - 1e-12:
                best_delta = delta
                best_target = int(neighbor_site_id)
        if best_target is None:
            return MorphogenesisActionRequest("wait", actor_site_id)
        return MorphogenesisActionRequest("swap", actor_site_id, best_target)


class RandomAdjacentSwapPolicy:
    """Local no-target adjacent-swap null control."""

    policy_id = "s07_random_adjacent_swap_control"
    policy_family = "e05_no_target_random_control"
    declared_information_access = "actor site and adjacent-neighbor occupancy only; no target identity or global state"

    def propose(
        self,
        *,
        state: SubstrateState,
        target: TargetMorphology,
        actor_site_id: int,
        rng: random.Random,
    ) -> MorphogenesisActionRequest:
        del target
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


def runnable_policy_specs() -> tuple[dict[str, Any], ...]:
    """Return S07 runnable controls with explicit access labels."""

    specs = []
    for policy in (LocalTargetNeighborDescentPolicy(), RandomAdjacentSwapPolicy()):
        specs.append(
            {
                "policy_id": policy.policy_id,
                "source_experiment": "E05",
                "source_path": "src/e05/scrambled_embryo.py",
                "policy_family": policy.policy_family,
                "source_policy_count": 1,
                "mapping_status": "runnable_control",
                "declared_information_access": policy.declared_information_access,
                "mapped_2d_observation": "S01 local substrate observation plus the access listed in declared_information_access",
                "blocker": "",
                "notes": "Runnable S07 control; not claimed as an E03/E04 transferred policy.",
            }
        )
    return tuple(specs)


def policy_by_id(policy_id: str) -> ScrambledPolicy:
    """Instantiate one runnable S07 control policy."""

    if policy_id == LocalTargetNeighborDescentPolicy.policy_id:
        return LocalTargetNeighborDescentPolicy()
    if policy_id == RandomAdjacentSwapPolicy.policy_id:
        return RandomAdjacentSwapPolicy()
    raise KeyError(f"unsupported S07 policy_id: {policy_id}")


def two_dimensional_targets(targets: Sequence[TargetMorphology] | None = None) -> tuple[TargetMorphology, ...]:
    """Return S03 gallery targets that are genuine 2D scrambled-embryo targets."""

    selected = []
    for target in targets or default_target_gallery():
        dimensions = target.substrate.dimensions
        if target.substrate.kind.endswith("_2d") and int(dimensions.get("height", 1)) > 1:
            selected.append(target)
    return tuple(selected)


def target_coverage_rows(targets: Sequence[TargetMorphology] | None = None) -> list[dict[str, Any]]:
    """Document which S03 targets are simulated by the S07 2D benchmark."""

    rows = []
    for target in targets or default_target_gallery():
        is_2d = target.substrate.kind.endswith("_2d") and int(target.substrate.dimensions.get("height", 1)) > 1
        rows.append(
            {
                "target_id": target.target_id,
                "target_kind": target.target_kind,
                "substrate_kind": target.substrate.kind,
                "site_count": len(target.substrate.site_ids),
                "simulated_in_s07": bool(is_2d),
                "status": "simulated_2d_target" if is_2d else "documented_not_simulated",
                "reason": (
                    "Genuine 2D S03 target."
                    if is_2d
                    else "The S03 sorted_row target is a 1D continuity target already covered by S06, not a 2D scrambled-embryo morphology."
                ),
            }
        )
    return rows


def scramble_target_state(target: TargetMorphology, seed: int) -> SubstrateState:
    """Return a deterministic severe permutation of a constructed target state."""

    rng = random.Random(int(seed))
    source = target.constructed_substrate()
    site_ids = tuple(source.site_ids)
    cells = [source.cell_at(site_id) for site_id in site_ids]
    if any(cell is None for cell in cells):
        raise ValueError("constructed target state unexpectedly has empty sites")
    rng.shuffle(cells)
    state = target.substrate.copy_empty()
    state.fill_sites(tuple(cells), site_ids)
    if target.target_error(state) == 0.0 and len(site_ids) > 1:
        cells = list(cells)
        cells[0], cells[1] = cells[1], cells[0]
        state = target.substrate.copy_empty()
        state.fill_sites(tuple(cells), site_ids)
    return state


def site_identity_error(target: TargetMorphology, state: SubstrateState, site_id: int) -> float:
    """Return target identity error at a single site."""

    cell = state.cell_at(int(site_id))
    if cell is None:
        return 1.0
    try:
        observed = identity_from_substrate_cell(cell, fallback_scalar=False)
    except ValueError:
        return 1.0
    return float(target.schema.distance(observed, target.identities_by_site[int(site_id)]))


def adjacent_swap_target_error_delta(target: TargetMorphology, state: SubstrateState, left_site_id: int, right_site_id: int) -> float:
    """Return proposed-minus-current local identity error for one adjacent swap."""

    left_site_id = int(left_site_id)
    right_site_id = int(right_site_id)
    left_cell = state.cell_at(left_site_id)
    right_cell = state.cell_at(right_site_id)
    if left_cell is None or right_cell is None:
        return 0.0
    try:
        left_identity = identity_from_substrate_cell(left_cell, fallback_scalar=False)
        right_identity = identity_from_substrate_cell(right_cell, fallback_scalar=False)
    except ValueError:
        return 0.0
    current = (
        target.schema.distance(left_identity, target.identities_by_site[left_site_id])
        + target.schema.distance(right_identity, target.identities_by_site[right_site_id])
    )
    proposed = (
        target.schema.distance(right_identity, target.identities_by_site[left_site_id])
        + target.schema.distance(left_identity, target.identities_by_site[right_site_id])
    )
    return float(proposed - current)


def scramble_displacement_fraction(target: TargetMorphology, state: SubstrateState) -> float:
    """Fraction of sites whose occupant has nonzero target identity error."""

    site_ids = tuple(target.substrate.site_ids)
    if not site_ids:
        return 0.0
    displaced = sum(1 for site_id in site_ids if site_identity_error(target, state, site_id) > 1e-12)
    return float(displaced / len(site_ids))


def compute_state_metrics(target: TargetMorphology, state: SubstrateState, state_label: str) -> dict[str, Any]:
    """Evaluate S05 metrics and return target plus aggregate errors."""

    metrics = evaluate_morphology_metrics(target, state, state_label=state_label)
    return {
        "target_error": float(target.target_error(state)),
        "aggregate_morphospace_error": float(aggregate_morphospace_error(metrics)),
        "metric_rows": metric_result_rows(metrics),
    }


def simulate_scrambled_run(
    *,
    target: TargetMorphology,
    policy_id: str,
    seed: int,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
) -> ScrambledRunResult:
    """Run one scrambled-target trajectory using S04 actions and S05 metrics."""

    state = scramble_target_state(target, seed)
    initial_state = clone_substrate_state(state)
    policy = policy_by_id(policy_id)
    executor = ActionExecutor()
    rng = random.Random(int(seed) * 104_729 + stable_int_hash(f"{target.target_id}:{policy_id}"))
    site_ids = tuple(target.substrate.site_ids)
    event_cap = max(1, int(event_multiplier) * len(site_ids))
    record_every = max(1, event_cap // max(1, int(records_per_run)))
    initial_metrics = compute_state_metrics(target, state, "initial")
    trace_rows = [
        _trace_row(
            target=target,
            policy=policy,
            seed=seed,
            event_step=0,
            accepted_swaps=0,
            attempted_swaps=0,
            rejected_actions=0,
            energy_cost=0.0,
            target_error=initial_metrics["target_error"],
            aggregate_morphospace_error=initial_metrics["aggregate_morphospace_error"],
        )
    ]
    metric_rows = _tag_metric_rows(initial_metrics["metric_rows"], target=target, policy=policy, seed=seed, state_label="initial")

    accepted_swaps = 0
    attempted_swaps = 0
    rejected_actions = 0
    wait_actions = 0
    total_energy = 0.0
    population_initial = len(site_ids)
    population_min = population_initial
    population_max = population_initial
    final_metrics = initial_metrics
    for event_step in range(1, event_cap + 1):
        actor_site_id = rng.choice(site_ids)
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
                    policy=policy,
                    seed=seed,
                    event_step=event_step,
                    accepted_swaps=accepted_swaps,
                    attempted_swaps=attempted_swaps,
                    rejected_actions=rejected_actions,
                    energy_cost=total_energy,
                    target_error=final_metrics["target_error"],
                    aggregate_morphospace_error=final_metrics["aggregate_morphospace_error"],
                )
            )
    final_metrics = compute_state_metrics(target, state, "final")
    metric_rows.extend(_tag_metric_rows(final_metrics["metric_rows"], target=target, policy=policy, seed=seed, state_label="final"))
    if trace_rows[-1]["event_step"] != event_cap:
        trace_rows.append(
            _trace_row(
                target=target,
                policy=policy,
                seed=seed,
                event_step=event_cap,
                accepted_swaps=accepted_swaps,
                attempted_swaps=attempted_swaps,
                rejected_actions=rejected_actions,
                energy_cost=total_energy,
                target_error=final_metrics["target_error"],
                aggregate_morphospace_error=final_metrics["aggregate_morphospace_error"],
            )
        )
    else:
        trace_rows[-1]["target_error"] = final_metrics["target_error"]
        trace_rows[-1]["aggregate_morphospace_error"] = final_metrics["aggregate_morphospace_error"]

    initial_target_error = float(initial_metrics["target_error"])
    final_target_error = float(final_metrics["target_error"])
    initial_aggregate = float(initial_metrics["aggregate_morphospace_error"])
    final_aggregate = float(final_metrics["aggregate_morphospace_error"])
    summary_row = {
        "research_step_id": "S07",
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "target_hash": target.target_hash,
        "substrate_kind": target.substrate.kind,
        "site_count": len(site_ids),
        "policy_id": policy.policy_id,
        "policy_family": policy.policy_family,
        "declared_information_access": policy.declared_information_access,
        "simulation_seed": int(seed),
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
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "wait_actions": int(wait_actions),
        "total_energy_cost": float(total_energy),
        "population_initial": int(population_initial),
        "population_final": sum(1 for site_id in site_ids if state.cell_at(site_id) is not None),
        "population_min": int(population_min),
        "population_max": int(population_max),
        "population_delta_total": sum(1 for site_id in site_ids if state.cell_at(site_id) is not None) - population_initial,
    }
    return ScrambledRunResult(
        summary_row=summary_row,
        trace_rows=tuple(trace_rows),
        metric_rows=tuple(metric_rows),
        initial_state=initial_state,
        final_state=clone_substrate_state(state),
    )


def run_scrambled_benchmark(
    *,
    targets: Sequence[TargetMorphology] | None = None,
    policy_ids: Sequence[str] = RUNNABLE_POLICY_IDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    event_multiplier: int = DEFAULT_EVENT_MULTIPLIER,
    records_per_run: int = DEFAULT_RECORDS_PER_RUN,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, int], ScrambledRunResult]]:
    """Run the complete compact S07 benchmark matrix."""

    selected_targets = two_dimensional_targets(targets)
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    run_results: dict[tuple[str, str, int], ScrambledRunResult] = {}
    for target in selected_targets:
        for policy_id in policy_ids:
            for seed in seeds:
                result = simulate_scrambled_run(
                    target=target,
                    policy_id=policy_id,
                    seed=int(seed),
                    event_multiplier=event_multiplier,
                    records_per_run=records_per_run,
                )
                summary_rows.append(result.summary_row)
                trace_rows.extend(result.trace_rows)
                metric_rows.extend(result.metric_rows)
                run_results[(target.target_id, policy_id, int(seed))] = result
    return summary_rows, trace_rows, metric_rows, run_results


def policy_source_mapping_rows(paths: Sequence[Path] = POLICY_SOURCE_PATHS) -> list[dict[str, Any]]:
    """Inventory E03/E04 policy artifacts and record 2D mapping decisions."""

    rows = [dict(row) for row in runnable_policy_specs()]
    for path in paths:
        path = Path(path)
        source_experiment = "E04" if "/E04/" in str(path) or path.name.startswith("e04_") else "E03"
        source_count, sample_ids = count_policy_records(path)
        policy_family = _policy_family_from_path(path)
        blocker = _mapping_blocker_for_source(path, source_experiment)
        rows.append(
            {
                "policy_id": policy_family,
                "source_experiment": source_experiment,
                "source_path": str(path),
                "policy_family": policy_family,
                "source_policy_count": int(source_count),
                "mapping_status": "blocked_without_expanding_information_access",
                "declared_information_access": _declared_access_for_source(path, source_experiment),
                "mapped_2d_observation": "",
                "blocker": blocker,
                "sample_policy_ids_json": json.dumps(sample_ids, sort_keys=True),
                "notes": (
                    "No S07 2D execution row was produced for this source because the available policy contract does not define a valid "
                    "four-neighbor 2D target-morphology observation/action mapping."
                ),
            }
        )
    return rows


def count_policy_records(path: Path) -> tuple[int, list[str]]:
    """Best-effort policy count and sample IDs for JSON, JSONL, and Markdown sources."""

    if not path.exists():
        return 0, []
    if path.suffix == ".jsonl":
        count = 0
        samples: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if len(samples) < 5:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    samples.append(_record_id(record, fallback=f"jsonl_line_{count}"))
        return count, samples
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("policies", "policyRecords", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return len(value), [_record_id(record, fallback=f"{key}_{idx}") for idx, record in enumerate(value[:5])]
        if isinstance(payload, list):
            return len(payload), [_record_id(record, fallback=f"json_{idx}") for idx, record in enumerate(payload[:5])]
        return 1, []
    if path.suffix in {".md", ".txt"}:
        return 1, [path.stem]
    return 1, []


def write_trace_zarr(trace_df: Any, output_path: Path) -> None:
    """Write trace rows as a compact zarr group with categorical string codes."""

    import pandas as pd
    import zarr

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trace_df).reset_index(drop=True)
    root = zarr.open_group(str(output_path), mode="w")
    root.attrs["schema"] = "eidosoma.e05_s07.scrambled_embryo_traces.zarr.v1"
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


def validate_trace_zarr(output_path: Path, expected_rows: int) -> dict[str, Any]:
    """Validate that the zarr trace group is readable and row-aligned."""

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


def recovery_fraction(initial_error: float, final_error: float) -> float:
    if initial_error <= 0.0:
        return 1.0 if final_error <= 0.0 else 0.0
    return float((initial_error - final_error) / initial_error)


def clone_substrate_state(state: SubstrateState) -> SubstrateState:
    clone = state.copy_empty()
    for site_id in state.site_ids:
        cell = state.cell_at(site_id)
        if cell is not None:
            clone.place_cell(site_id, cell)
    return clone


def stable_int_hash(text: str) -> int:
    value = 0
    for char in str(text):
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value


def _trace_row(
    *,
    target: TargetMorphology,
    policy: ScrambledPolicy,
    seed: int,
    event_step: int,
    accepted_swaps: int,
    attempted_swaps: int,
    rejected_actions: int,
    energy_cost: float,
    target_error: float,
    aggregate_morphospace_error: float,
) -> dict[str, Any]:
    return {
        "research_step_id": "S07",
        "target_id": target.target_id,
        "target_kind": target.target_kind,
        "policy_id": policy.policy_id,
        "policy_family": policy.policy_family,
        "simulation_seed": int(seed),
        "event_step": int(event_step),
        "accepted_swaps": int(accepted_swaps),
        "attempted_swaps": int(attempted_swaps),
        "rejected_actions": int(rejected_actions),
        "energy_cost": float(energy_cost),
        "target_error": float(target_error),
        "aggregate_morphospace_error": float(aggregate_morphospace_error),
    }


def _tag_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: TargetMorphology,
    policy: ScrambledPolicy,
    seed: int,
    state_label: str,
) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        payload = dict(row)
        payload.update(
            {
                "research_step_id": "S07",
                "target_id": target.target_id,
                "target_kind": target.target_kind,
                "policy_id": policy.policy_id,
                "policy_family": policy.policy_family,
                "simulation_seed": int(seed),
                "benchmark_state_label": state_label,
            }
        )
        tagged.append(payload)
    return tagged


def _record_id(record: Any, fallback: str) -> str:
    if isinstance(record, Mapping):
        for key in ("policy_id", "policyId", "policyName", "id"):
            if key in record:
                return str(record[key])
    return fallback


def _policy_family_from_path(path: Path) -> str:
    name = path.name
    if "classic" in name:
        return "e03_classic_1d_policy_library"
    if "frontier" in name:
        return "e03_frontier_candidate_1d_policies"
    if "generated" in name:
        return "e03_generated_1d_rule_library"
    if "qd" in name:
        return "e03_qd_discovered_1d_policies"
    if "evolved_repair" in name:
        return "e04_evolved_local_memory_signal_repair_policies"
    if "local_learning_rule" in name:
        return "e04_local_learning_rule_spec"
    return path.stem


def _declared_access_for_source(path: Path, source_experiment: str) -> str:
    name = path.name
    if source_experiment == "E04" and "evolved_repair" in name:
        return (
            "src.e04.no_oracle_protocol.LocalTrainingObservation: actor scalar value/status, left/right values/statuses, "
            "last-action memory, bounded counters, neighbor identity count, and allowed blocked/frustrated signals"
        )
    if source_experiment == "E04":
        return "E04 local learning and repair contracts over 1D adjacent left/right observations"
    if "classic" in name:
        return "E03 legacy PolicyObservation with actor scalar value, full 1D values/statuses, left/right indices, and ideal_position"
    return "E03 1D rule-DSL policy features over scalar values, sortedness-oriented left/right movement, and adjacent swap actions"


def _mapping_blocker_for_source(path: Path, source_experiment: str) -> str:
    name = path.name
    if source_experiment == "E04" and "evolved_repair" in name:
        return (
            "The E04 selected policies emit swap-left/swap-right decisions from a scalar LocalTrainingObservation. "
            "Mapping them to north/east/south/west 2D S03 targets would require an undeclared orientation/projection or target-morphology feature."
        )
    if source_experiment == "E04":
        return (
            "The E04 local learning rule is specified for 1D repair and scalar adjacent observations; S07 2D target recovery needs "
            "a declared four-neighbor observation/action mapping before execution."
        )
    if "classic" in name:
        return (
            "The classic E03 wrappers depend on 1D scalar sorted-order context, including full value/status arrays or ideal-position semantics. "
            "A 2D S03 target mapping would add global projection or target-position information."
        )
    return (
        "The E03 generated/discovered rule policies were selected under 1D sortedness semantics with scalar values and left/right actions. "
        "No declared feature maps their observations to 2D target morphology or chooses among four grid neighbors."
    )
