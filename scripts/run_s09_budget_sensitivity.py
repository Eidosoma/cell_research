#!/usr/bin/env python3
"""Extend every S09 fixed-horizon active arm to 32,768 opportunities."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.engine import evaluate_terminal, execute_batch
from reference_simulator.model import Architecture
from src.detours.barrier_interventions import (
    METRICS,
    _CoupledSchedule,
    _PulseInterceptor,
    _PulseProposalFactory,
    _temporary_activation_scenario,
    dynamic_state_digest,
    make_scenario,
    metric_levels,
    structural_state_from_ordinal,
)
from src.detours.state_space import FamilySpec, StructuralState


ROOT = Path("/artifacts/research_steps/S09")
S04 = Path("/artifacts/research_steps/S04")
EXTENDED_BUDGET = 32_768


def load_families() -> dict[int, FamilySpec]:
    inventory = pq.read_table(S04 / "state_family_inventory.parquet").to_pandas()
    return {
        int(row.family_ordinal): FamilySpec.from_canonical_dict(
            json.loads(row.canonical_family_json)
        )
        for row in inventory.itertuples(index=False)
    }


def summary_run(task: Mapping[str, Any]) -> dict[str, Any]:
    family = FamilySpec.from_canonical_dict(task["family"])
    source = task["row"]
    focal_index = int(str(source["focal_barrier_id"])[1:])
    state = structural_state_from_ordinal(
        family, int(source["source_state_ordinal"])
    ).to_run_state()
    scenario = make_scenario(
        family,
        seed=int(source["coupling_seed"]),
        max_activations=EXTENDED_BUDGET,
        generation_key=(
            f"E03-S09-sensitivity:{source['source_family_ordinal']}:"
            f"{source['source_state_ordinal']}:{source['replicate_index']}:"
            f"{source['arm_variant']}"
        ),
    )
    state.terminal = evaluate_terminal(scenario, state)
    levels = metric_levels(scenario, state)
    starts = dict(levels)
    peaks = dict(levels)
    pulse = source["intervention_type"] == "activate"
    proposal_factory = None
    interceptor = None
    pulse_focal = None
    if pulse:
        pulse_focal = str(source["focal_barrier_id"])
        temporary = _temporary_activation_scenario(
            family,
            focal_index,
            seed=int(source["coupling_seed"]),
            max_activations=EXTENDED_BUDGET,
            generation_key=scenario.generation_key,
        )
        proposal_factory = _PulseProposalFactory(temporary, pulse_focal)
        interceptor = _PulseInterceptor(pulse_focal)
    schedule = None
    if family.architecture == Architecture.CELL_VIEW:
        schedule = _CoupledSchedule(
            scenario,
            coupling_key=str(source["coupling_key"]),
            seed=int(source["coupling_seed"]),
            pulse_focal_id=pulse_focal,
        )
    prefix_hash = None
    prefix_ledger = None
    prefix_levels = None
    while state.terminal is None:
        execute_batch(
            scenario,
            state,
            retain_events=False,
            emit_event_records=False,
            schedule_factory=schedule,
            proposal_factory=proposal_factory,
            execution_interceptor=interceptor,
        )
        levels = metric_levels(scenario, state)
        for metric in METRICS:
            peaks[metric] = max(peaks[metric], levels[metric])
        if state.activation_count == int(source["event_budget"]):
            prefix_hash = dynamic_state_digest(state)
            prefix_ledger = dict(state.ledger)
            prefix_levels = dict(levels)
    if prefix_hash is None or prefix_ledger is None or prefix_levels is None:
        raise RuntimeError("extended run did not cross the primary horizon")
    final_structural = StructuralState.from_run_state(family, state)
    result = {
        "schema_version": "e03.s09.event_budget_sensitivity.v1",
        "research_step_id": "S09",
        "primary_run_id": source["run_id"],
        "pair_block_id": source["pair_block_id"],
        "source_family_ordinal": int(source["source_family_ordinal"]),
        "source_state_ordinal": int(source["source_state_ordinal"]),
        "arm_family_ordinal": int(source["arm_family_ordinal"]),
        "intervention_type": source["intervention_type"],
        "arm_variant": source["arm_variant"],
        "n": int(source["n"]),
        "architecture": source["architecture"],
        "primary_budget": int(source["event_budget"]),
        "extended_budget": EXTENDED_BUDGET,
        "primary_prefix_dynamic_equal": prefix_hash
        == source["final_dynamic_state_sha256"],
        "primary_prefix_ledger_equal": all(
            int(prefix_ledger[key]) == int(source[f"cost_{key}"])
            for key in prefix_ledger
        ),
        "primary_prefix_final_metrics_equal": all(
            int(prefix_levels[metric]) == int(source[f"final_{metric}"])
            for metric in METRICS
        ),
        "extended_stop_reason": state.terminal,
        "extended_completed": state.terminal == "complete",
        "extended_event_budget_censored": state.terminal == "event_budget",
        "extended_event_count": state.activation_count,
        "extended_final_state_ordinal": final_structural.selection_cursor_code
        * math.factorial(family.n)
        + final_structural.occupancy_rank,
        "extended_final_dynamic_state_sha256": dynamic_state_digest(state),
        "extended_full_ledger_unit_cost": sum(state.ledger.values()),
    }
    for key, value in state.ledger.items():
        result[f"extended_cost_{key}"] = value
    for metric in METRICS:
        result[f"start_{metric}"] = starts[metric]
        result[f"extended_peak_{metric}"] = peaks[metric]
        result[f"extended_excursion_{metric}"] = peaks[metric] - starts[metric]
        result[f"extended_final_{metric}"] = levels[metric]
    return result


def main() -> None:
    results = pq.read_table(ROOT / "barrier_interventions.parquet").to_pandas()
    censored = results[results.event_budget_censored].copy()
    if len(censored) != 3749:
        raise RuntimeError(f"primary event-budget count drifted: {len(censored)}")
    families = load_families()
    tasks = [
        {
            "row": row._asdict(),
            "family": families[int(row.arm_family_ordinal)].canonical_dict(),
        }
        for row in censored.itertuples(index=False)
    ]
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(summary_run, tasks, chunksize=4))
    frame = pd.DataFrame(rows)
    if len(frame) != len(censored):
        raise RuntimeError("event-budget sensitivity lost a primary arm")
    if not (
        frame.primary_prefix_dynamic_equal.all()
        and frame.primary_prefix_ledger_equal.all()
        and frame.primary_prefix_final_metrics_equal.all()
    ):
        raise RuntimeError("extended run disagrees with its primary 2,048-event prefix")
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": b"e03.s09.event_budget_sensitivity.v1", b"researchStepId": b"S09"}
    )
    pq.write_table(
        table,
        ROOT / "event_budget_sensitivity.parquet",
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
    )
    summary = {
        "primaryEventBudgetArms": len(frame),
        "completedByExtendedHorizon": int(frame.extended_completed.sum()),
        "stillActiveAtExtendedHorizon": int(frame.extended_event_budget_censored.sum()),
        "prefixValidationPassed": True,
        "workers": workers,
    }
    (ROOT / "event_budget_sensitivity_summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
