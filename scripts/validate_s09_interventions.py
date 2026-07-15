#!/usr/bin/env python3
"""Independent full-corpus validation for E03 S09 artifacts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.detours.barrier_interventions import METRICS, execute_intervention_arm
from src.detours.state_space import FamilySpec


ROOT = Path("/artifacts/research_steps/S09")
S04 = Path("/artifacts/research_steps/S04")
S07 = Path("/artifacts/research_steps/S07")
CLASS_LABELS = {
    0: "complete_start",
    1: "reachable_no_detour",
    2: "necessary_detour",
    3: "unreachable_active",
    4: "quiescent",
}


def load_families() -> dict[int, FamilySpec]:
    inventory = pq.read_table(S04 / "state_family_inventory.parquet").to_pandas()
    return {
        int(row.family_ordinal): FamilySpec.from_canonical_dict(
            json.loads(row.canonical_family_json)
        )
        for row in inventory.itertuples(index=False)
    }


def replay_one(task: Mapping[str, Any]) -> dict[str, Any]:
    source = task["row"]
    family = FamilySpec.from_canonical_dict(task["family"])
    target = source["target_cell_id"]
    replay, _ = execute_intervention_arm(
        family,
        int(source["source_state_ordinal"]),
        source_family_ordinal=int(source["source_family_ordinal"]),
        arm_family_ordinal=int(source["arm_family_ordinal"]),
        intervention_type=str(source["intervention_type"]),
        arm_variant=str(source["arm_variant"]),
        focal_index=int(str(source["focal_barrier_id"])[1:]),
        target_index=(int(str(target)[1:]) if target is not None and pd.notna(target) else None),
        replicate_index=int(source["replicate_index"]),
        coupling_key=str(source["coupling_key"]),
        coupling_seed=int(source["coupling_seed"]),
        seed_search_attempts=int(source["seed_search_attempts"]),
        max_activations=int(source["event_budget"]),
    )
    fields = [
        "stop_reason", "completed", "event_budget_censored", "event_count",
        "accepted_action_count", "pulse_delivered", "final_state_ordinal",
        "final_dynamic_state_sha256", "event_digest_sha256", "compact_trace_sha256",
        "actor_token_prefix_sha256", "side_token_prefix_sha256", "trace_complete",
        "ledger_identities_valid", "full_ledger_unit_cost",
    ]
    fields.extend(
        f"cost_{name}"
        for name in (
            "activations", "observationReads", "valueComparisons", "proposals",
            "noOps", "rejections", "memoryUpdates", "acceptedSwaps",
            "displacedCells", "conflictLosses",
        )
    )
    for metric in METRICS:
        fields.extend(
            [
                f"start_{metric}", f"peak_{metric}", f"excursion_{metric}",
                f"final_{metric}", f"worsening_events_{metric}",
            ]
        )
    mismatches = [
        field
        for field in fields
        if replay[field] != source[field]
        and not (pd.isna(replay[field]) and pd.isna(source[field]))
    ]
    return {
        "run_id": source["run_id"],
        "source_family_ordinal": int(source["source_family_ordinal"]),
        "source_state_ordinal": int(source["source_state_ordinal"]),
        "intervention_type": source["intervention_type"],
        "replay_match": not mismatches,
        "mismatched_fields": json.dumps(mismatches),
    }


def write_parquet(frame: pd.DataFrame, name: str, schema: str) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S09"}
    )
    pq.write_table(
        table, ROOT / name, compression="zstd", compression_level=9,
        use_dictionary=True, write_statistics=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-replay",
        action="store_true",
        help="Reuse a complete replay_validation.parquet produced by this validator",
    )
    args = parser.parse_args()
    results = pq.read_table(ROOT / "barrier_interventions.parquet").to_pandas()
    pairs = pq.read_table(ROOT / "paired_effects.parquet").to_pandas()
    metric_effects = pq.read_table(ROOT / "paired_metric_effects.parquet").to_pandas()
    coverage = pq.read_table(ROOT / "necessary_start_coverage.parquet").to_pandas()
    optimum = pq.read_table(ROOT / "structural_optimum_changes.parquet").to_pandas()
    isolation = pq.read_table(ROOT / "intervention_isolation_audit.parquet").to_pandas()
    stream = pq.read_table(ROOT / "random_stream_audit.parquet").to_pandas()
    sensitivity = pq.read_table(ROOT / "event_budget_sensitivity.parquet").to_pandas()
    families = load_families()

    gates: dict[str, bool] = {}
    required = [
        "barrier_interventions.parquet", "paired_effects.parquet",
        "paired_metric_effects.parquet", "effect_summary.parquet",
        "necessary_start_coverage.parquet", "structural_optimum_changes.parquet",
        "intervention_isolation_audit.parquet", "random_stream_audit.parquet",
        "event_budget_sensitivity.parquet", "paired_traces.parquet",
        "barrier_effects.png", "input_immutability.json", "environment.json",
        "build_summary.json",
    ]
    gates["required_outputs_present_nonempty"] = all(
        (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0 for name in required
    )
    unique_starts = coverage[["family_ordinal", "state_ordinal"]].drop_duplicates()
    start_sizes = unique_starts.merge(
        coverage[["family_ordinal", "state_ordinal", "n"]].drop_duplicates(),
        on=["family_ordinal", "state_ordinal"],
    ).groupby("n").size().to_dict()
    gates["necessary_start_coverage"] = (
        len(unique_starts) == 407
        and start_sizes == {4: 371, 5: 36}
        and coverage.family_ordinal.nunique() == 59
        and len(coverage) == 1_628
        and int((coverage.classification_code == 2).sum()) == 1_107
        and coverage.selected_for_s09.all()
    )
    expected_runs = {
        "baseline": 3_116,
        "remove": 3_116,
        "activate": 3_096,
        "move": 9_496,
        "add": 9_496,
    }
    gates["run_and_pair_accounting"] = (
        len(results) == 28_320
        and results.intervention_type.value_counts().to_dict() == expected_runs
        and results.run_id.nunique() == len(results)
        and results.pair_block_id.nunique() == 3_116
        and len(pairs) == 25_204
        and len(metric_effects) == 100_816
    )
    gates["prestate_and_isolation"] = (
        pairs.pre_dynamic_state_equal.all()
        and isolation.isolated.all()
        and (isolation.pre_dynamic_state_hash_count == 1).all()
    )
    cell_stream = stream[stream.architecture == "cell_view"]
    gates["common_random_stream"] = (
        cell_stream.actor_token_common_prefix_equal.all()
        and cell_stream.side_token_common_prefix_equal.all()
        and len(cell_stream) > 0
    )
    gates["pulse_delivery_and_traditional_boundary"] = (
        len(results[results.intervention_type == "activate"]) == 3_096
        and results.loc[
            results.intervention_type == "activate", "pulse_delivered"
        ].fillna(False).all()
        and not (
            (results.architecture == "traditional")
            & (results.intervention_type == "activate")
        ).any()
        and coverage.loc[
            coverage.architecture == "traditional",
            "activation_intervention_defined",
        ].eq(False).all()
    )
    gates["trace_ledger_and_primary_horizon"] = (
        results.trace_complete.all()
        and results.ledger_identities_valid.all()
        and int(results.event_budget_censored.sum()) == 3_749
        and (results.event_count <= 2_048).all()
    )
    gates["event_budget_sensitivity"] = (
        len(sensitivity) == 3_749
        and set(sensitivity.primary_run_id)
        == set(results.loc[results.event_budget_censored, "run_id"])
        and sensitivity.primary_prefix_dynamic_equal.all()
        and sensitivity.primary_prefix_ledger_equal.all()
        and sensitivity.primary_prefix_final_metrics_equal.all()
        and (sensitivity.extended_event_count <= 32_768).all()
    )

    # Recompute every paired scalar and metric delta from the arm summaries.
    base = results[results.intervention_type == "baseline"].set_index("run_id")
    arms = results.set_index("run_id")
    pair_ok = True
    for row in pairs.itertuples(index=False):
        left = base.loc[row.baseline_run_id]
        right = arms.loc[row.arm_run_id]
        pair_ok &= int(right.completed) - int(left.completed) == int(row.delta_completed)
        pair_ok &= int(right.cost_activations) - int(left.cost_activations) == int(row.delta_activations)
        pair_ok &= int(right.full_ledger_unit_cost) - int(left.full_ledger_unit_cost) == int(row.delta_full_ledger_unit_cost)
        for metric in METRICS:
            pair_ok &= int(right[f"peak_{metric}"]) - int(left[f"peak_{metric}"]) == int(getattr(row, f"delta_peak_{metric}"))
            pair_ok &= int(right[f"excursion_{metric}"]) - int(left[f"excursion_{metric}"]) == int(getattr(row, f"delta_excursion_{metric}"))
    gates["paired_effect_recomputation"] = bool(pair_ok)

    # Exact independent S07 join for every persistent intervention start.
    arm_families = sorted(int(value) for value in optimum.arm_family_ordinal.unique())
    paths = ds.dataset(S07 / "path_solutions.parquet", format="parquet").to_table(
        filter=ds.field("family_ordinal").isin(arm_families)
    ).to_pandas()
    joined = optimum.merge(
        paths,
        left_on=["arm_family_ordinal", "arm_state_ordinal", "metric_code"],
        right_on=["family_ordinal", "state_ordinal", "metric_code"],
        how="left",
        validate="many_to_one",
    )
    exact_ok = len(joined) == len(optimum) == 13_312
    exact_ok &= all(
        CLASS_LABELS[int(row.classification_code)] == row.arm_classification
        and int(row.minimum_peak) == int(row.arm_minimum_peak)
        and int(row.minimum_excursion) == int(row.arm_minimum_excursion)
        for row in joined.itertuples(index=False)
    )
    safe = (
        optimum.arm_goal_reachable
        == optimum.arm_classification.isin(
            ["complete_start", "reachable_no_detour", "necessary_detour"]
        )
    ).all()
    safe &= optimum.loc[~optimum.arm_goal_reachable, "delta_minimum_excursion"].isna().all()
    gates["exact_s07_join_and_reachability_safe_effects"] = bool(exact_ok and safe)

    workers = min(8, os.cpu_count() or 1)
    replay_path = ROOT / "replay_validation.parquet"
    if args.reuse_replay:
        replay = pq.read_table(replay_path).to_pandas()
        if set(replay.run_id) != set(results.run_id) or len(replay) != len(results):
            raise RuntimeError("replay reuse requested but the saved key set is incomplete")
    else:
        tasks = [
            {
                "row": row._asdict(),
                "family": families[int(row.arm_family_ordinal)].canonical_dict(),
            }
            for row in results.itertuples(index=False)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            replay_rows = list(pool.map(replay_one, tasks, chunksize=8))
        replay = pd.DataFrame(replay_rows)
        write_parquet(replay, "replay_validation.parquet", "e03.s09.replay_validation.v1")
    gates["all_run_deterministic_replay"] = len(replay) == 28_320 and replay.replay_match.all()

    immutable = json.loads((ROOT / "input_immutability.json").read_text())
    gates["input_immutability"] = bool(immutable["allUnchanged"])
    gates["no_s10_artifacts"] = not Path("/artifacts/research_steps/S10").exists()
    gates = {name: bool(result) for name, result in gates.items()}
    passed = all(gates.values())
    value = {
        "schemaVersion": "e03.s09.validation_results.v1",
        "researchStepId": "S09",
        "passed": passed,
        "gateCount": len(gates),
        "passedGateCount": sum(gates.values()),
        "gates": gates,
        "workers": workers,
        "replayedRuns": len(replay),
    }
    (ROOT / "validation_results.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    lines = [f"{name}: {'PASS' if result else 'FAIL'}" for name, result in gates.items()]
    (ROOT / "validation.log").write_text("\n".join(lines) + "\n")
    print(json.dumps(value, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
