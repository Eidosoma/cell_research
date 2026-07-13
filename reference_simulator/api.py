"""Stable Python analysis API, exact replay, and bounded replicate parallelism."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Literal, Sequence

from .engine import run
from .model import Architecture, Cell, Direction, FaultMode, Policy, RunResult, Scenario
from .rng import permutation


REFERENCE_SEMANTICS_VERSION = "E01-reference-v1"


def create_scenario(
    values: Sequence[int | float],
    *,
    policy: Policy | str,
    architecture: Architecture | str = Architecture.CELL_VIEW,
    direction: Direction | str = Direction.ASCENDING,
    seed: int = 0,
    max_activations: int = 100_000,
    generation_key: str,
    permute: bool = True,
    faults: dict[int, FaultMode | str] | None = None,
    fault_count: int | None = None,
    fault_mode: FaultMode | str = FaultMode.PASSIVE,
    batch_width: int = 1,
) -> Scenario:
    """Create a homogeneous-policy scenario with an explicit pre-ID generation key."""
    selected_policy = Policy(policy)
    selected_architecture = Architecture(architecture)
    selected_direction = Direction(direction)
    if faults is not None and fault_count is not None:
        raise ValueError("provide explicit faults or fault_count, not both")
    if fault_count is not None and not 0 <= fault_count <= len(values):
        raise ValueError("fault_count must be in [0, n]")
    explicit_faults = dict(faults or {})
    if any(not 0 <= index < len(values) for index in explicit_faults):
        raise ValueError("explicit fault index is outside the scenario")
    if any(FaultMode(mode) == FaultMode.NORMAL for mode in explicit_faults.values()):
        raise ValueError("explicit fault entries must be passive or stuck")
    ids = tuple(f"cell-{index:04d}" for index in range(len(values)))
    fault_placement = "explicit"
    if fault_count is not None:
        if FaultMode(fault_mode) == FaultMode.NORMAL and fault_count:
            raise ValueError("fault_mode must be passive or stuck when fault_count is positive")
        selected = set(permutation(ids, seed, generation_key + "/fault-placement")[:fault_count])
        explicit_faults = {
            index: FaultMode(fault_mode) for index, cell_id in enumerate(ids) if cell_id in selected
        }
        fault_placement = "reference_without_replacement"
    cells = tuple(
        Cell(
            cell_id=f"cell-{index:04d}",
            value=value,
            policy=selected_policy,
            direction=selected_direction,
            fault=FaultMode(explicit_faults.get(index, FaultMode.NORMAL)),
        )
        for index, value in enumerate(values)
    )
    occupancy = permutation(ids, seed, generation_key + "/occupancy") if permute else ids
    return Scenario.create(
        cells,
        initial_occupancy=occupancy,
        seed=seed,
        max_activations=max_activations,
        architecture=selected_architecture,
        batch_width=batch_width,
        traditional_policy=selected_policy if selected_architecture == Architecture.TRADITIONAL else None,
        generation_key=generation_key,
        fault_placement=fault_placement,
        requested_fault_count=(fault_count if fault_count is not None else len(explicit_faults)),
    )


def run_scenario(
    scenario: Scenario,
    *,
    trace_mode: Literal["full", "digest", "none"] = "digest",
) -> RunResult:
    return run(scenario, trace_mode=trace_mode)


def exact_replay(result: RunResult) -> RunResult:
    """Rerun a result's scenario and require byte-identical stable serialization."""
    replayed = run(result.scenario, trace_mode=result.summary["traceMode"])
    if replayed.to_json_bytes() != result.to_json_bytes():
        raise AssertionError("exact replay mismatch")
    return replayed


def _run_worker(item: tuple[Scenario, str]) -> bytes:
    scenario, trace_mode = item
    return run(scenario, trace_mode=trace_mode).to_json_bytes()


def run_many(
    scenarios: Sequence[Scenario],
    *,
    workers: int = 1,
    trace_mode: Literal["full", "digest", "none"] = "digest",
) -> list[RunResult]:
    """Run independent replicates, preserving input order across 1–8 workers."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    if workers == 1:
        return [run(scenario, trace_mode=trace_mode) for scenario in scenarios]
    items = [(scenario, trace_mode) for scenario in scenarios]
    # ``spawn`` avoids unsafe fork-from-multithreaded-host behavior and does not
    # affect counter-addressed outputs.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        payloads = list(executor.map(_run_worker, items))
    return [RunResult.from_json_bytes(payload) for payload in payloads]
