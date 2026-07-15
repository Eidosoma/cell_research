#!/usr/bin/env python3
"""Independent scientific and handoff validation for E03 S10."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from src.detours.action_suppression import (
    METRIC_DELTA_COLUMNS,
    execute_suppression_arm_summary,
)
from src.detours.barrier_interventions import METRICS, execute_intervention_arm
from src.detours.state_space import FamilySpec


OUTPUT = Path("/artifacts/research_steps/S10")
CACHE = Path("/cache/e03_s10")
S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
S09 = Path("/artifacts/research_steps/S09")
REPOSITORY = Path(__file__).resolve().parents[1]
WORKERS = min(8, os.cpu_count() or 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def reverse_reachable(
    node_count: int,
    sources: np.ndarray,
    targets: np.ndarray,
    goals: np.ndarray,
) -> np.ndarray:
    predecessors: list[list[int]] = [[] for _ in range(node_count)]
    for source, target in zip(sources.tolist(), targets.tolist()):
        predecessors[target].append(source)
    reached = np.zeros(node_count, dtype=bool)
    queue: deque[int] = deque()
    for goal in goals.tolist():
        reached[goal] = True
        queue.append(goal)
    while queue:
        node = queue.popleft()
        for predecessor in predecessors[node]:
            if not reached[predecessor]:
                reached[predecessor] = True
                queue.append(predecessor)
    return reached


def validate_independent_reachability(reach: pd.DataFrame) -> int:
    family_ordinals = sorted(int(value) for value in reach.arm_family_ordinal.unique())
    edge_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("edge_shard_*.parquet")), format="parquet"
    )
    node_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("node_shard_*.parquet")), format="parquet"
    )
    edges = edge_dataset.to_table(
        columns=[
            "family_ordinal",
            "source_state_ordinal",
            "successor_state_ordinal",
            "decision_code",
            *METRIC_DELTA_COLUMNS.values(),
        ],
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    nodes = node_dataset.to_table(
        columns=["family_ordinal", "state_ordinal", "terminal_code"],
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    comparisons = 0
    for family_ordinal in family_ordinals:
        family_edges = edges[edges.family_ordinal == family_ordinal]
        family_nodes = nodes[nodes.family_ordinal == family_ordinal].sort_values(
            "state_ordinal"
        )
        node_count = len(family_nodes)
        terminal = family_nodes.terminal_code.to_numpy(dtype=np.uint8)
        goals = np.flatnonzero(terminal == 1)
        contexts = reach[reach.arm_family_ordinal == family_ordinal]
        for (metric, threshold), labels in contexts.groupby(
            ["filter_metric", "filter_threshold"], sort=True
        ):
            suppressed = family_edges.decision_code.eq(1).to_numpy() & (
                family_edges[METRIC_DELTA_COLUMNS[metric]].to_numpy()
                > int(threshold)
            )
            reached = reverse_reachable(
                node_count,
                family_edges.source_state_ordinal.to_numpy(dtype=np.int64)[~suppressed],
                family_edges.successor_state_ordinal.to_numpy(dtype=np.int64)[~suppressed],
                goals,
            )
            for row in labels.itertuples(index=False):
                state = int(row.arm_state_ordinal)
                expected_status = (
                    "quiescent"
                    if terminal[state] == 2
                    else ("reachable" if reached[state] else "unreachable_active")
                )
                require(
                    row.filtered_exact_status == expected_status,
                    f"independent filtered reach mismatch {family_ordinal}/{state}/{metric}/{threshold}",
                )
                comparisons += 1
    return comparisons


def replay_source_task(
    task: tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    runs, family_payloads, audit_by_run = task
    families = {
        ordinal: FamilySpec.from_canonical_dict(payload)
        for ordinal, payload in family_payloads.items()
    }
    rows: list[dict[str, Any]] = []
    compare_fields = [
        "scenario_id",
        "pre_dynamic_state_sha256",
        "initial_terminal",
        "stop_reason",
        "completed",
        "event_budget_censored",
        "event_count",
        "accepted_action_count",
        "final_state_ordinal",
        "final_dynamic_state_sha256",
        "native_event_prefix_sha256",
        "native_event_prefix_count",
        "compact_trace_sha256",
        "run_summary_sha256",
        "actor_token_prefix_sha256",
        "side_token_prefix_sha256",
        "side_consumption_prefix_sha256",
        "trace_complete",
        "summary_complete",
        "ledger_identities_valid",
        "full_ledger_unit_cost",
    ]
    for expected in runs:
        family = families[int(expected["arm_family_ordinal"])]
        target = expected.get("target_cell_id")
        target_index = None if pd.isna(target) else int(str(target)[1:])
        (
            actual,
            _,
            filter_,
            _,
            _,
            _,
        ) = execute_suppression_arm_summary(
            family,
            int(expected["source_state_ordinal"]),
            source_family_ordinal=int(expected["source_family_ordinal"]),
            arm_family_ordinal=int(expected["arm_family_ordinal"]),
            intervention_type=str(expected["intervention_type"]),
            arm_variant=str(expected["arm_variant"]),
            focal_index=int(str(expected["focal_barrier_id"])[1:]),
            target_index=target_index,
            replicate_index=int(expected["replicate_index"]),
            coupling_key=str(expected["coupling_key"]),
            coupling_seed=int(expected["coupling_seed"]),
            seed_search_attempts=int(expected["seed_search_attempts"]),
            maximum_activations=2_048,
            metric=str(expected["filter_metric"]),
            threshold=int(expected["filter_threshold"]),
        )
        fields = list(compare_fields)
        fields.extend(column for column in expected if column.startswith("cost_"))
        fields.extend(
            column
            for column in expected
            if column.startswith(
                ("start_", "peak_", "excursion_", "final_", "worsening_events_")
            )
        )
        result_match = all(actual[field] == expected[field] for field in fields)
        actual_audit = filter_.audit_row()
        expected_audit = audit_by_run[expected["run_id"]]
        audit_fields = [
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
        ]
        audit_match = all(
            (actual_audit[field] == expected_audit[field])
            or (pd.isna(actual_audit[field]) and pd.isna(expected_audit[field]))
            for field in audit_fields
        )
        rows.append(
            {
                "schema_version": "e03.s10.replay_validation.v1",
                "research_step_id": "S10",
                "run_id": expected["run_id"],
                "source_family_ordinal": int(expected["source_family_ordinal"]),
                "source_state_ordinal": int(expected["source_state_ordinal"]),
                "filter_metric": expected["filter_metric"],
                "filter_threshold": int(expected["filter_threshold"]),
                "result_match": result_match,
                "filter_audit_match": audit_match,
                "replay_match": result_match and audit_match,
            }
        )
    return rows


def deterministic_replay(
    results: pd.DataFrame, audits: pd.DataFrame
) -> pd.DataFrame:
    filtered = results[results.condition == "metric_filtered"].copy()
    inventory = pd.read_parquet(S04 / "state_family_inventory.parquet").set_index(
        "family_ordinal"
    )
    audit_by_run = audits.set_index("run_id").to_dict("index")
    tasks = []
    for _, group in filtered.groupby(
        ["source_family_ordinal", "source_state_ordinal"], sort=True
    ):
        payloads = {
            int(ordinal): json.loads(
                inventory.loc[int(ordinal), "canonical_family_json"]
            )
            for ordinal in group.arm_family_ordinal.unique()
        }
        selected_audits = {
            run_id: audit_by_run[run_id] for run_id in group.run_id.tolist()
        }
        tasks.append((group.to_dict("records"), payloads, selected_audits))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for task_rows in executor.map(replay_source_task, tasks, chunksize=1):
            rows.extend(task_rows)
    frame = pd.DataFrame(rows).sort_values("run_id")
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"schemaVersion": b"e03.s10.replay_validation.v1",
            b"researchStepId": b"S10",
        }
    )
    pq.write_table(
        table,
        OUTPUT / "replay_validation.parquet",
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        version="2.6",
    )
    return frame


def completed_replay_or_rerun(
    results: pd.DataFrame, audits: pd.DataFrame
) -> pd.DataFrame:
    """Reuse only a complete, exact-ID replay artifact after a reporting failure.

    The default always performs the exhaustive replay.  The opt-in recovery path
    exists for a failure after ``deterministic_replay`` has atomically written its
    complete artifact; it validates schema, row count, uniqueness, exact run-ID
    coverage, and all match flags before accepting that completed computation.
    """

    if os.environ.get("S10_REUSE_COMPLETED_REPLAY") != "1":
        return deterministic_replay(results, audits)
    path = OUTPUT / "replay_validation.parquet"
    require(path.is_file(), "completed replay recovery artifact exists")
    replay = pd.read_parquet(path)
    expected_ids = set(
        results.loc[results.condition.eq("metric_filtered"), "run_id"].tolist()
    )
    require(len(replay) == 114_208, "completed replay recovery row count")
    require(replay.run_id.nunique() == len(replay), "completed replay unique IDs")
    require(set(replay.run_id.tolist()) == expected_ids, "completed replay exact IDs")
    require(
        replay[["result_match", "filter_audit_match", "replay_match"]]
        .all()
        .all(),
        "completed replay recovery match flags",
    )
    return replay


def main() -> None:
    required = [
        "action_suppression_results.parquet",
        "filter_audit.parquet",
        "paired_effects.parquet",
        "filtered_structural_reachability.parquet",
        "effect_summary.parquet",
        "event_budget_sensitivity.parquet",
        "representative_traces.parquet",
        "suppression_effects.png",
    ]
    require(all((OUTPUT / name).is_file() for name in required), "required S10 outputs")
    results = pd.read_parquet(OUTPUT / "action_suppression_results.parquet")
    audits = pd.read_parquet(OUTPUT / "filter_audit.parquet")
    pairs = pd.read_parquet(OUTPUT / "paired_effects.parquet")
    reach = pd.read_parquet(OUTPUT / "filtered_structural_reachability.parquet")
    budget = pd.read_parquet(OUTPUT / "event_budget_sensitivity.parquet")
    gates: dict[str, bool] = {}
    gates["complete_corpus_accounting"] = (
        len(results) == 139_432
        and results.condition.eq("unfiltered_control").sum() == 25_224
        and results.condition.eq("metric_filtered").sum() == 114_208
        and len(audits) == len(pairs) == 114_208
        and pairs.filter_threshold.eq(0).sum() == 100_896
        and pairs.filter_threshold.eq(1).sum() == 13_312
    )
    gates["source_context_family_coverage"] = (
        pairs[["source_family_ordinal", "source_state_ordinal"]].drop_duplicates().shape[0]
        == 407
        and reach[
            [
                "source_family_ordinal",
                "source_state_ordinal",
                "arm_family_ordinal",
                "arm_state_ordinal",
                "arm_variant",
            ]
        ].drop_duplicates().shape[0]
        == 3_328
        and reach.arm_family_ordinal.nunique() == 394
        and len(reach) == 26_624
    )
    gates["filter_correctness"] = (
        audits.event_partition_valid.all()
        and audits.eligible_partition_valid.all()
        and audits.filter_decision_exact.all()
        and audits.false_positive_count.eq(0).all()
        and audits.false_negative_count.eq(0).all()
        and audits.audited_event_count.eq(audits.trace_event_count).all()
    )
    gates["pairing_and_predivergence"] = all(
        pairs[field].all()
        for field in (
            "pre_dynamic_state_equal",
            "scenario_id_equal",
            "actor_token_common_prefix_equal",
            "side_token_common_prefix_equal",
            "side_consumption_common_prefix_equal",
            "first_divergence_equals_first_suppression",
            "pre_divergence_identity",
        )
    ) and (
        pairs.loc[pairs.suppressed_count > 0, "divergence_proposal_equal"].all()
        and pairs.loc[pairs.suppressed_count > 0, "divergence_prestate_equal"].all()
        and pairs.loc[
            pairs.suppressed_count > 0, "divergence_decision_is_filter_only"
        ].all()
        and pairs.loc[
            pairs.suppressed_count == 0, "no_suppression_full_trace_equal"
        ].all()
    )
    comparisons = validate_independent_reachability(reach)
    gates["independent_filtered_reachability"] = comparisons == 26_624
    gates["impossibility_censoring_separation"] = (
        not (
            pairs.intervention_created_impossibility & pairs.completion_censored
        ).any()
        and not (
            pairs.exact_impossibility_not_censoring & pairs.completion_censored
        ).any()
    )
    gates["failure_cost_not_efficiency"] = (
        pairs.loc[
            ~pairs.efficiency_comparable,
            [
                "efficiency_delta_activations",
                "efficiency_delta_accepted_swaps",
                "efficiency_delta_full_ledger_unit_cost",
            ],
        ]
        .isna()
        .all()
        .all()
    )
    executed = budget.sensitivity_action.eq("extended_exact_reachable")
    gates["event_budget_sensitivity"] = (
        not budget.empty
        and all(
            budget.loc[executed, field].all()
            for field in (
                "primary_prefix_dynamic_equal",
                "primary_prefix_ledger_equal",
                "primary_prefix_final_metrics_equal",
                "primary_prefix_filter_audit_equal",
                "primary_prefix_stream_counters_equal",
            )
        )
        and budget.loc[~executed, "filtered_exact_status"].ne("reachable").all()
    )
    replay = completed_replay_or_rerun(results, audits)
    gates["deterministic_replay"] = (
        len(replay) == 114_208 and replay.replay_match.all()
    )
    before = json.loads((CACHE / "input_hashes_before.json").read_text())
    after = {path: sha256_file(Path(path)) for path in before}
    unchanged = {path: before[path] == after[path] for path in before}
    gates["input_immutability"] = all(unchanged.values())
    write_json(
        OUTPUT / "input_immutability.json",
        {
            "schemaVersion": "e03.s10.input_immutability.v1",
            "researchStepId": "S10",
            "inputs": [
                {
                    "path": path,
                    "beforeSha256": before[path],
                    "afterSha256": after[path],
                    "unchanged": unchanged[path],
                }
                for path in sorted(before)
            ],
            "allUnchanged": all(unchanged.values()),
        },
    )
    gates["no_s11_artifacts"] = not Path("/artifacts/research_steps/S11").exists()
    # pandas/numpy reductions can return np.bool_; normalize the handoff payload
    # to builtin booleans so canonical JSON serialization remains deterministic.
    gates = {name: bool(value) for name, value in gates.items()}
    success = all(gates.values())
    validation = {
        "schemaVersion": "e03.s10.validation_results.v1",
        "researchStepId": "S10",
        "success": success,
        "gates": gates,
        "gateCount": len(gates),
        "gatesPassed": sum(gates.values()),
        "independentReachabilityComparisons": comparisons,
        "deterministicReplays": len(replay),
        "deterministicReplayMatches": int(replay.replay_match.sum()),
    }
    write_json(OUTPUT / "validation_results.json", validation)
    (OUTPUT / "validation.log").write_text(
        "\n".join(f"{'PASS' if value else 'FAIL'} {key}" for key, value in gates.items())
        + "\n"
    )
    print(json.dumps(validation, sort_keys=True))
    if not success:
        raise RuntimeError("S10 scientific validation failed")


if __name__ == "__main__":
    main()
