#!/usr/bin/env python3
"""Run the frozen S10 32,768-opportunity sensitivity population."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import Architecture, FaultMode, Policy, canonical_json_bytes
from src.detours.action_suppression import execute_extended_suppression_summary
from src.detours.barrier_interventions import METRICS, coupled_token
from src.detours.state_space import FamilySpec


OUTPUT = Path("/artifacts/research_steps/S10")
S04 = Path("/artifacts/research_steps/S04")
WORKERS = min(8, os.cpu_count() or 1)
PRIMARY_BUDGET = 2_048
EXTENDED_BUDGET = 32_768


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError("hard stop: empty S10 event-budget sensitivity")
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"schemaVersion": b"e03.s10.event_budget_sensitivity.v1",
            b"researchStepId": b"S10",
        }
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def expected_stream_counters(
    family: FamilySpec, seed: int, coupling_key: str, count: int
) -> dict[str, int]:
    if family.architecture == Architecture.TRADITIONAL:
        return {}
    actor_blocks = 0
    side_count = 0
    for event_index in range(count):
        token = coupled_token(seed, coupling_key, event_index, family.n)
        actor_blocks += token.actor_blocks
        actor = token.actor_index
        if (
            family.policies[actor] == Policy.BUBBLE
            and family.faults[actor] == FaultMode.NORMAL
        ):
            side_count += 1
    result = {"actor_activation": actor_blocks}
    if side_count:
        result["bubble_side"] = side_count
    return result


def run_candidate(task: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, audit, family_payload = task
    family = FamilySpec.from_canonical_dict(family_payload)
    extended = execute_extended_suppression_summary(
        family,
        int(row["source_state_ordinal"]),
        source_family_ordinal=int(row["source_family_ordinal"]),
        replicate_index=int(row["replicate_index"]),
        arm_variant=str(row["arm_variant"]),
        coupling_key=str(row["coupling_key"]),
        coupling_seed=int(row["coupling_seed"]),
        metric=str(row["filter_metric"]),
        threshold=int(row["filter_threshold"]),
        maximum_activations=EXTENDED_BUDGET,
        prefix_activation=PRIMARY_BUDGET,
    )
    prefix = extended.pop("prefix")
    if prefix is None:
        raise RuntimeError(f"hard stop: missing S10 prefix capture {row['run_id']}")
    expected_ledger = {
        key.removeprefix("cost_"): int(value)
        for key, value in row.items()
        if key.startswith("cost_")
    }
    expected_levels = {name: int(row[f"final_{name}"]) for name in METRICS}
    expected_filter = {
        key: audit[key]
        for key in prefix["prefix_filter_audit"]
        if key in audit
    }
    comparable_filter_fields = set(expected_filter) - {
        "first_suppression_proposal_sha256"
    }
    expected_streams = expected_stream_counters(
        family, int(row["coupling_seed"]), str(row["coupling_key"]), PRIMARY_BUDGET
    )
    result = {
        "schema_version": "e03.s10.event_budget_sensitivity.v1",
        "research_step_id": "S10",
        "filtered_run_id": row["run_id"],
        "s09_control_run_id": row["s09_control_run_id"],
        "source_family_ordinal": int(row["source_family_ordinal"]),
        "source_state_ordinal": int(row["source_state_ordinal"]),
        "arm_family_ordinal": int(row["arm_family_ordinal"]),
        "arm_state_ordinal": int(row["arm_state_ordinal"]),
        "arm_variant": row["arm_variant"],
        "intervention_type": row["intervention_type"],
        "filter_metric": row["filter_metric"],
        "filter_threshold": int(row["filter_threshold"]),
        "replicate_index": int(row["replicate_index"]),
        "filtered_exact_status": row["filtered_exact_status"],
        "sensitivity_action": "extended_exact_reachable",
        "primary_budget": PRIMARY_BUDGET,
        "extended_budget": EXTENDED_BUDGET,
        "primary_prefix_dynamic_equal": prefix["prefix_dynamic_state_sha256"]
        == row["final_dynamic_state_sha256"],
        "primary_prefix_ledger_equal": prefix["prefix_ledger"] == expected_ledger,
        "primary_prefix_final_metrics_equal": prefix["prefix_levels"] == expected_levels,
        "primary_prefix_filter_audit_equal": all(
            prefix["prefix_filter_audit"][key] == expected_filter[key]
            for key in comparable_filter_fields
        ),
        "primary_prefix_filter_digest_recording_boundary": (
            "native-prefix proposal digest includes random-draw metadata; "
            "summary extension proposal digest omits event-record draws"
        ),
        "primary_prefix_stream_counters_equal": prefix["prefix_stream_counters"]
        == expected_streams,
        **extended,
    }
    return result


def main() -> None:
    results = pd.read_parquet(OUTPUT / "action_suppression_results.parquet")
    pairs = pd.read_parquet(OUTPUT / "paired_effects.parquet")
    audits = pd.read_parquet(OUTPUT / "filter_audit.parquet")
    inventory = pd.read_parquet(S04 / "state_family_inventory.parquet").set_index(
        "family_ordinal"
    )
    candidates = results[
        results.condition.eq("metric_filtered")
        & results.filter_threshold.eq(0)
        & results.replicate_index.eq(0)
        & results.stop_reason.eq("event_budget")
    ].merge(
        pairs[
            ["filtered_run_id", "filtered_exact_status"]
        ],
        left_on="run_id",
        right_on="filtered_run_id",
        how="left",
        validate="one_to_one",
    ).merge(
        audits,
        on="run_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_audit"),
    )
    if candidates.empty or candidates.filtered_exact_status.isna().any():
        raise RuntimeError("hard stop: S10 primary budget population is empty/incomplete")
    reachable = candidates[candidates.filtered_exact_status == "reachable"].copy()
    impossible = candidates[candidates.filtered_exact_status != "reachable"].copy()
    tasks = []
    for row in reachable.to_dict("records"):
        family_payload = json.loads(
            inventory.loc[int(row["arm_family_ordinal"]), "canonical_family_json"]
        )
        audit = {
            key.removesuffix("_audit"): value
            for key, value in row.items()
            if key.endswith("_audit")
        }
        audit["filter_metric"] = row["filter_metric"]
        audit["filter_threshold"] = int(row["filter_threshold"])
        # Merge columns that were unique to the audit and therefore unsuffixed.
        for key in (
            "audited_event_count",
            "native_eligible_count",
            "native_ineligible_count",
            "eligible_improving_count",
            "eligible_neutral_count",
            "eligible_allowed_worsening_count",
            "eligible_excess_worsening_count",
            "suppressed_count",
            "false_positive_count",
            "false_negative_count",
            "event_partition_valid",
            "eligible_partition_valid",
            "filter_decision_exact",
            "first_suppression_event",
            "first_suppression_delta",
            "first_suppression_before",
            "first_suppression_candidate_after",
            "first_suppression_proposal_sha256",
            "maximum_suppressed_delta",
            "total_suppressed_delta",
        ):
            audit[key] = None if pd.isna(row[key]) else row[key]
        tasks.append((row, audit, family_payload))
    extended_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for result in executor.map(run_candidate, tasks, chunksize=1):
            extended_rows.append(result)
    for row in impossible.to_dict("records"):
        extended_rows.append(
            {
                "schema_version": "e03.s10.event_budget_sensitivity.v1",
                "research_step_id": "S10",
                "filtered_run_id": row["run_id"],
                "s09_control_run_id": row["s09_control_run_id"],
                "source_family_ordinal": int(row["source_family_ordinal"]),
                "source_state_ordinal": int(row["source_state_ordinal"]),
                "arm_family_ordinal": int(row["arm_family_ordinal"]),
                "arm_state_ordinal": int(row["arm_state_ordinal"]),
                "arm_variant": row["arm_variant"],
                "intervention_type": row["intervention_type"],
                "filter_metric": row["filter_metric"],
                "filter_threshold": int(row["filter_threshold"]),
                "replicate_index": int(row["replicate_index"]),
                "filtered_exact_status": row["filtered_exact_status"],
                "sensitivity_action": "not_extended_exact_impossibility",
                "primary_budget": PRIMARY_BUDGET,
                "extended_budget": EXTENDED_BUDGET,
                "primary_prefix_dynamic_equal": None,
                "primary_prefix_ledger_equal": None,
                "primary_prefix_final_metrics_equal": None,
                "primary_prefix_filter_audit_equal": None,
                "primary_prefix_stream_counters_equal": None,
                "extended_stop_reason": None,
                "extended_completed": None,
                "extended_event_budget_censored": None,
                "extended_event_count": None,
                "extended_final_state_ordinal": None,
                "extended_final_dynamic_state_sha256": None,
                "extended_full_ledger_unit_cost": None,
                "extended_ledger_identities_valid": None,
            }
        )
    frame = pd.DataFrame(extended_rows).sort_values("filtered_run_id")
    if len(frame) != len(candidates):
        raise RuntimeError("hard stop: event-budget sensitivity coverage mismatch")
    executed = frame.sensitivity_action.eq("extended_exact_reachable")
    prefix_fields = [
        "primary_prefix_dynamic_equal",
        "primary_prefix_ledger_equal",
        "primary_prefix_final_metrics_equal",
        "primary_prefix_filter_audit_equal",
        "primary_prefix_stream_counters_equal",
    ]
    if not all(frame.loc[executed, field].all() for field in prefix_fields):
        raise RuntimeError("hard stop: an extended S10 prefix differs at 2,048")
    write_parquet(OUTPUT / "event_budget_sensitivity.parquet", frame)
    summary = {
        "schemaVersion": "e03.s10.event_budget_sensitivity_summary.v1",
        "researchStepId": "S10",
        "primaryReplicate0BudgetRuns": len(candidates),
        "exactReachableExtended": int(executed.sum()),
        "exactImpossibleNotExtended": int((~executed).sum()),
        "extendedCompleted": int(frame.loc[executed, "extended_completed"].sum()),
        "extendedStillActive": int(
            frame.loc[executed, "extended_stop_reason"].eq("event_budget").sum()
        ),
        "extendedQuiescent": int(
            frame.loc[executed, "extended_stop_reason"].eq("quiescent").sum()
        ),
    }
    write_json(OUTPUT / "event_budget_sensitivity_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
