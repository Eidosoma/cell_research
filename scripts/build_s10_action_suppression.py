#!/usr/bin/env python3
"""Build the exhaustive paired E03 S10 action-suppression corpus."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from reference_simulator.model import Architecture, canonical_json_bytes
from src.detours.action_suppression import (
    METRIC_DELTA_COLUMNS,
    MetricWorseningFilter,
    canonical_digest,
    compare_suppression_pair,
    execute_suppression_arm_summary,
    filtered_class_label,
    solve_filtered_graph,
)
from src.detours.barrier_interventions import METRICS, execute_intervention_arm
from src.detours.necessary_detour import CLASS_LABELS
from src.detours.state_space import FamilySpec


OUTPUT = Path("/artifacts/research_steps/S10")
CACHE = Path("/cache/e03_s10")
S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
S06 = Path("/artifacts/research_steps/S06")
S07 = Path("/artifacts/research_steps/S07")
S08 = Path("/artifacts/research_steps/S08")
S09 = Path("/artifacts/research_steps/S09")
REPOSITORY = Path(__file__).resolve().parents[1]
WORKERS = min(8, os.cpu_count() or 1)
PRIMARY_THRESHOLD = 0
SENSITIVITY_THRESHOLD = 1
BOOTSTRAPS = 10_000
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
TRACE_EXTRA_FIELDS = {
    "proposal_actor_pos",
    "proposal_target_pos",
    "proposal_new_cursor",
    "proposal_observation_reads",
    "proposal_value_comparisons",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    schema_version: str,
) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"refusing to write empty S10 table: {path.name}")
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema_version.encode(), b"researchStepId": b"S10"}
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


def input_paths() -> list[Path]:
    paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        S04 / "research_step_full_results.md",
        S04 / "state_family_inventory.parquet",
        S05 / "research_step_full_results.md",
        S05 / "graph_corpus_manifest.json",
        S05 / "graph_family_manifest.parquet",
        S06 / "research_step_full_results.md",
        S06 / "necessary_detour_spec.md",
        S07 / "research_step_full_results.md",
        S07 / "path_solutions.parquet",
        S08 / "research_step_full_results.md",
        S08 / "behavior_necessity_comparison.parquet",
        S09 / "research_step_full_results.md",
        S09 / "barrier_interventions.parquet",
        S09 / "structural_optimum_changes.parquet",
        S09 / "event_budget_sensitivity.parquet",
        REPOSITORY / "analysis/s10_action_suppression_contract.json",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
        REPOSITORY / "src/detours/action_suppression.py",
        REPOSITORY / "src/detours/barrier_interventions.py",
    ]
    paths.extend(sorted(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md")))
    return paths


def load_families() -> tuple[pd.DataFrame, dict[int, FamilySpec]]:
    inventory = pd.read_parquet(S04 / "state_family_inventory.parquet")
    families = {
        int(row.family_ordinal): FamilySpec.from_canonical_dict(
            json.loads(row.canonical_family_json)
        )
        for row in inventory.itertuples(index=False)
    }
    return inventory, families


def load_controls() -> pd.DataFrame:
    controls = pd.read_parquet(S09 / "barrier_interventions.parquet")
    controls = controls[
        controls.intervention_type.isin(["baseline", "remove", "move", "add"])
    ].copy()
    controls = controls.sort_values(
        [
            "source_family_ordinal",
            "source_state_ordinal",
            "arm_variant",
            "replicate_index",
        ]
    ).reset_index(drop=True)
    if len(controls) != 25_224:
        raise RuntimeError(f"hard stop: expected 25,224 controls, found {len(controls)}")
    contexts = controls[controls.replicate_index == 0][
        [
            "source_family_ordinal",
            "source_state_ordinal",
            "arm_family_ordinal",
            "arm_state_ordinal",
            "arm_variant",
        ]
    ].drop_duplicates()
    if len(contexts) != 3_328:
        raise RuntimeError(f"hard stop: expected 3,328 contexts, found {len(contexts)}")
    if len(controls[["source_family_ordinal", "source_state_ordinal"]].drop_duplicates()) != 407:
        raise RuntimeError("hard stop: S10 source-start coverage is not 407")
    if controls.arm_family_ordinal.nunique() != 394:
        raise RuntimeError("hard stop: S10 arm-family coverage is not 394")

    sensitivity = pd.read_parquet(S09 / "event_budget_sensitivity.parquet")
    persistent_budget = controls[controls.stop_reason == "event_budget"]
    joined = persistent_budget[["run_id"]].merge(
        sensitivity[["primary_run_id", "extended_stop_reason", "extended_event_count"]],
        left_on="run_id",
        right_on="primary_run_id",
        how="left",
        validate="one_to_one",
    )
    if len(joined) != 3_194 or joined.extended_stop_reason.isna().any():
        raise RuntimeError("hard stop: persistent S09 recurrent sensitivity is incomplete")
    if not (
        joined.extended_stop_reason.eq("event_budget").all()
        and joined.extended_event_count.eq(32_768).all()
    ):
        raise RuntimeError("hard stop: an S09 persistent budget run is not recurrent at 32,768")
    controls["s09_observed_status"] = np.select(
        [controls.completed, controls.stop_reason.eq("quiescent")],
        ["completed", "quiescent"],
        default="recurrent_active_32768",
    )
    return controls


def _status_from_class(label: str) -> str:
    if label in {"complete_start", "reachable_no_detour", "necessary_detour"}:
        return "reachable"
    return label


def build_filtered_reachability(
    controls: pd.DataFrame,
) -> pd.DataFrame:
    contexts = controls[controls.replicate_index == 0][
        [
            "source_family_ordinal",
            "source_state_ordinal",
            "arm_family_ordinal",
            "arm_state_ordinal",
            "n",
            "architecture",
            "direction",
            "policy_profile",
            "intervention_type",
            "arm_variant",
            "focal_barrier_id",
            "target_cell_id",
        ]
    ].drop_duplicates()
    family_ordinals = sorted(int(value) for value in contexts.arm_family_ordinal.unique())
    edge_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("edge_shard_*.parquet")), format="parquet"
    )
    node_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("node_shard_*.parquet")), format="parquet"
    )
    edge_columns = [
        "family_ordinal",
        "source_state_ordinal",
        "successor_state_ordinal",
        "decision_code",
        *METRIC_DELTA_COLUMNS.values(),
    ]
    edges = edge_dataset.to_table(
        columns=edge_columns,
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    nodes = node_dataset.to_table(
        columns=["family_ordinal", "state_ordinal", "terminal_code"],
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    paths = ds.dataset(S07 / "path_solutions.parquet", format="parquet").to_table(
        columns=[
            "family_ordinal",
            "state_ordinal",
            "metric_code",
            "metric_level",
            "classification_code",
            "minimum_peak",
            "minimum_excursion",
        ],
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    graph_manifest = pd.read_parquet(S05 / "graph_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    rows: list[dict[str, Any]] = []
    for family_ordinal in family_ordinals:
        graph = graph_manifest.loc[family_ordinal]
        node_count = int(graph.node_count)
        family_edges = edges[edges.family_ordinal == family_ordinal].sort_values(
            ["source_state_ordinal", "successor_state_ordinal"], kind="stable"
        )
        family_nodes = nodes[nodes.family_ordinal == family_ordinal].sort_values(
            "state_ordinal"
        )
        if (
            len(family_nodes) != node_count
            or not np.array_equal(
                family_nodes.state_ordinal.to_numpy(), np.arange(node_count)
            )
            or len(family_edges) != int(graph.edge_count)
        ):
            raise RuntimeError(f"hard stop: graph slice mismatch for family {family_ordinal}")
        family_contexts = contexts[contexts.arm_family_ordinal == family_ordinal]
        for metric in METRICS:
            metric_code = METRIC_CODES[metric]
            labels = paths[
                (paths.family_ordinal == family_ordinal)
                & (paths.metric_code == metric_code)
            ].sort_values("state_ordinal")
            if len(labels) != node_count or not np.array_equal(
                labels.state_ordinal.to_numpy(), np.arange(node_count)
            ):
                raise RuntimeError(
                    f"hard stop: S07 level slice mismatch {family_ordinal}/{metric}"
                )
            for threshold in (PRIMARY_THRESHOLD, SENSITIVITY_THRESHOLD):
                solution, suppressed = solve_filtered_graph(
                    node_count,
                    family_edges.source_state_ordinal.to_numpy(),
                    family_edges.successor_state_ordinal.to_numpy(),
                    family_edges.decision_code.to_numpy(),
                    family_edges[METRIC_DELTA_COLUMNS[metric]].to_numpy(),
                    labels.metric_level.to_numpy(),
                    family_nodes.terminal_code.to_numpy(),
                    threshold,
                )
                for context in family_contexts.itertuples(index=False):
                    state = int(context.arm_state_ordinal)
                    original_class = CLASS_LABELS[int(labels.classification_code.iloc[state])]
                    filtered_class = filtered_class_label(solution.classifications[state])
                    original_status = _status_from_class(original_class)
                    filtered_status = _status_from_class(filtered_class)
                    rows.append(
                        {
                            **context._asdict(),
                            "filter_metric": metric,
                            "filter_threshold": threshold,
                            "s09_unfiltered_classification": original_class,
                            "s09_exact_start_status": original_status,
                            "s09_minimum_peak": int(labels.minimum_peak.iloc[state]),
                            "s09_minimum_excursion": int(
                                labels.minimum_excursion.iloc[state]
                            ),
                            "filtered_classification": filtered_class,
                            "filtered_exact_status": filtered_status,
                            "filtered_minimum_peak": (
                                int(solution.bottleneck_levels[state])
                                if filtered_status == "reachable"
                                else None
                            ),
                            "filtered_minimum_excursion": (
                                int(solution.excursion_levels[state])
                                if filtered_status == "reachable"
                                else None
                            ),
                            "intervention_created_impossibility": original_status
                            == "reachable"
                            and filtered_status != "reachable",
                            "preexisting_exact_impossibility": original_status
                            != "reachable",
                            "graph_edge_count": len(family_edges),
                            "graph_native_accepted_edge_count": int(
                                family_edges.decision_code.eq(1).sum()
                            ),
                            "graph_suppressed_edge_count": int(suppressed.sum()),
                            "graph_kept_edge_count": int(len(suppressed) - suppressed.sum()),
                        }
                    )
    result = pd.DataFrame(rows).sort_values(
        [
            "source_family_ordinal",
            "source_state_ordinal",
            "arm_variant",
            "filter_metric",
            "filter_threshold",
        ]
    ).reset_index(drop=True)
    if len(result) != 3_328 * 4 * 2:
        raise RuntimeError(f"hard stop: exact filtered rows {len(result)} != 26,624")
    if result.filtered_exact_status.isna().any():
        raise RuntimeError("hard stop: null filtered reachability status")
    return result


def _legacy_trace_digest(trace: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest(
        [
            {key: value for key, value in row.items() if key not in TRACE_EXTRA_FIELDS}
            for row in trace
        ]
    )


def _control_replay_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
) -> bool:
    fields = [
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
        "event_digest_sha256",
        "actor_token_prefix_sha256",
        "side_token_prefix_sha256",
        "trace_complete",
        "ledger_identities_valid",
        "full_ledger_unit_cost",
    ]
    fields.extend(column for column in expected if column.startswith("cost_"))
    fields.extend(
        column
        for column in expected
        if column.startswith(("start_", "peak_", "excursion_", "final_", "worsening_events_"))
    )
    return all(actual[field] == expected[field] for field in fields) and (
        _legacy_trace_digest(trace) == expected["compact_trace_sha256"]
    )


def _s10_control_row(row: Mapping[str, Any], s09: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    control_id = "s10c:" + canonical_digest({"s09RunId": s09["run_id"]})
    result.update(
        {
            "schema_version": "e03.s10.action_suppression_result.v1",
            "research_step_id": "S10",
            "run_id": control_id,
            "s09_control_run_id": s09["run_id"],
            "condition": "unfiltered_control",
            "filter_metric": None,
            "filter_threshold": None,
            "analysis_tier": "shared_control",
            "control_run_id": None,
            "s09_observed_status": s09["s09_observed_status"],
            "control_replay_matches_s09": True,
        }
    )
    return result


def _s10_filtered_row(
    row: Mapping[str, Any],
    s09: Mapping[str, Any],
    metric: str,
    threshold: int,
    control_id: str,
) -> dict[str, Any]:
    run_id = "s10f:" + canonical_digest(
        {
            "s09RunId": s09["run_id"],
            "metric": metric,
            "threshold": threshold,
        }
    )
    result = dict(row)
    result.update(
        {
            "schema_version": "e03.s10.action_suppression_result.v1",
            "research_step_id": "S10",
            "run_id": run_id,
            "s09_control_run_id": s09["run_id"],
            "condition": "metric_filtered",
            "filter_metric": metric,
            "filter_threshold": threshold,
            "analysis_tier": "primary" if threshold == 0 else "threshold_sensitivity",
            "control_run_id": control_id,
            "s09_observed_status": s09["s09_observed_status"],
            "control_replay_matches_s09": None,
        }
    )
    return result


def run_source_task(
    task: tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    controls, family_payloads = task
    families = {
        ordinal: FamilySpec.from_canonical_dict(payload)
        for ordinal, payload in family_payloads.items()
    }
    result_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for s09 in controls:
        family = families[int(s09["arm_family_ordinal"])]
        focal = int(str(s09["focal_barrier_id"])[1:])
        target_value = s09.get("target_cell_id")
        target = (
            int(str(target_value)[1:]) if not pd.isna(target_value) else None
        )
        common = dict(
            source_family_ordinal=int(s09["source_family_ordinal"]),
            arm_family_ordinal=int(s09["arm_family_ordinal"]),
            intervention_type=str(s09["intervention_type"]),
            arm_variant=str(s09["arm_variant"]),
            focal_index=focal,
            target_index=target,
            replicate_index=int(s09["replicate_index"]),
            coupling_key=str(s09["coupling_key"]),
            coupling_seed=int(s09["coupling_seed"]),
            seed_search_attempts=int(s09["seed_search_attempts"]),
            max_activations=2_048,
            trace_proposal_details=True,
        )
        control_row, control_trace = execute_intervention_arm(
            family, int(s09["source_state_ordinal"]), **common
        )
        if not _control_replay_matches(s09, control_row, control_trace):
            raise RuntimeError(f"hard stop: S09 control replay mismatch {s09['run_id']}")
        control_result = _s10_control_row(control_row, s09)
        result_rows.append(control_result)
        thresholds = [PRIMARY_THRESHOLD]
        if int(s09["replicate_index"]) == 0:
            thresholds.append(SENSITIVITY_THRESHOLD)
        for threshold in thresholds:
            for metric in METRICS:
                (
                    filtered_row,
                    filtered_trace,
                    filter_,
                    filtered_actor_tokens,
                    filtered_side_tokens,
                    filtered_side_consumption,
                ) = execute_suppression_arm_summary(
                    family,
                    int(s09["source_state_ordinal"]),
                    source_family_ordinal=int(s09["source_family_ordinal"]),
                    arm_family_ordinal=int(s09["arm_family_ordinal"]),
                    intervention_type=str(s09["intervention_type"]),
                    arm_variant=str(s09["arm_variant"]),
                    focal_index=focal,
                    target_index=target,
                    replicate_index=int(s09["replicate_index"]),
                    coupling_key=str(s09["coupling_key"]),
                    coupling_seed=int(s09["coupling_seed"]),
                    seed_search_attempts=int(s09["seed_search_attempts"]),
                    maximum_activations=2_048,
                    metric=metric,
                    threshold=threshold,
                )
                filtered_result = _s10_filtered_row(
                    filtered_row, s09, metric, threshold, control_result["run_id"]
                )
                audit = filter_.audit_row()
                audit.update(
                    {
                        "schema_version": "e03.s10.filter_audit.v1",
                        "research_step_id": "S10",
                        "run_id": filtered_result["run_id"],
                        "control_run_id": control_result["run_id"],
                        "s09_control_run_id": s09["run_id"],
                        "source_family_ordinal": int(s09["source_family_ordinal"]),
                        "source_state_ordinal": int(s09["source_state_ordinal"]),
                        "arm_family_ordinal": int(s09["arm_family_ordinal"]),
                        "arm_state_ordinal": int(s09["arm_state_ordinal"]),
                        "arm_variant": s09["arm_variant"],
                        "intervention_type": s09["intervention_type"],
                        "replicate_index": int(s09["replicate_index"]),
                        "trace_event_count": int(filtered_result["event_count"]),
                    }
                )
                pair = compare_suppression_pair(
                    control_result,
                    control_trace,
                    filtered_result,
                    filtered_trace,
                    audit,
                )
                control_actor_tokens = [
                    (int(item["event_index"]), str(item["actor_id"]))
                    for item in control_trace
                ]
                control_side_tokens = [
                    (int(item["event_index"]), item["raw_side_value"])
                    for item in control_trace
                ]
                control_side_consumption = [
                    (int(item["event_index"]), bool(item["side_consumed"]))
                    for item in control_trace
                ]
                full_common = min(
                    len(control_actor_tokens), len(filtered_actor_tokens)
                )
                traditional = str(s09["architecture"]) == "traditional"
                pair.update(
                    {
                        "common_prefix_event_count": full_common,
                        "actor_token_common_prefix_equal": traditional
                        or control_actor_tokens[:full_common]
                        == filtered_actor_tokens[:full_common],
                        "side_token_common_prefix_equal": traditional
                        or control_side_tokens[:full_common]
                        == filtered_side_tokens[:full_common],
                        "side_consumption_common_prefix_equal": traditional
                        or control_side_consumption[:full_common]
                        == filtered_side_consumption[:full_common],
                        "common_stream_applicable": not traditional,
                    }
                )
                pair.update(
                    {
                        "schema_version": "e03.s10.paired_effect.v1",
                        "research_step_id": "S10",
                        "pair_id": "s10p:"
                        + canonical_digest(
                            {
                                "control": control_result["run_id"],
                                "filtered": filtered_result["run_id"],
                            }
                        ),
                        "control_run_id": control_result["run_id"],
                        "filtered_run_id": filtered_result["run_id"],
                        "s09_control_run_id": s09["run_id"],
                        "source_family_ordinal": int(s09["source_family_ordinal"]),
                        "source_state_ordinal": int(s09["source_state_ordinal"]),
                        "arm_family_ordinal": int(s09["arm_family_ordinal"]),
                        "arm_state_ordinal": int(s09["arm_state_ordinal"]),
                        "n": int(s09["n"]),
                        "architecture": s09["architecture"],
                        "direction": s09["direction"],
                        "policy_profile": s09["policy_profile"],
                        "intervention_type": s09["intervention_type"],
                        "arm_variant": s09["arm_variant"],
                        "focal_barrier_id": s09["focal_barrier_id"],
                        "target_cell_id": (
                            None if pd.isna(s09.get("target_cell_id"))
                            else s09.get("target_cell_id")
                        ),
                        "replicate_index": int(s09["replicate_index"]),
                        "filter_metric": metric,
                        "filter_threshold": threshold,
                        "analysis_tier": (
                            "primary" if threshold == 0 else "threshold_sensitivity"
                        ),
                        "s09_observed_status": s09["s09_observed_status"],
                        "suppressed_count": int(audit["suppressed_count"]),
                    }
                )
                result_rows.append(filtered_result)
                audit_rows.append(audit)
                pair_rows.append(pair)
    return result_rows, audit_rows, pair_rows


def _pair_classification(row: pd.Series) -> str:
    if bool(row.intervention_created_impossibility):
        return "filter_created_exact_impossibility"
    if row.s09_exact_start_status != "reachable":
        return f"preexisting_exact_{row.s09_exact_start_status}"
    if row.control_completed and row.filtered_completed:
        return "both_complete"
    if row.control_completed and not row.filtered_completed:
        return "filter_failed_despite_exact_reachability"
    if not row.control_completed and row.filtered_completed:
        return "filter_rescued_control_failure"
    if row.filtered_stop_reason == "event_budget":
        return "both_or_filtered_noncomplete_reachable_censored"
    return "both_or_filtered_noncomplete_reachable_quiescent"


def join_structural_labels(
    pairs: pd.DataFrame, reachability: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "source_family_ordinal",
        "source_state_ordinal",
        "arm_family_ordinal",
        "arm_state_ordinal",
        "arm_variant",
        "filter_metric",
        "filter_threshold",
    ]
    columns = keys + [
        "s09_unfiltered_classification",
        "s09_exact_start_status",
        "s09_minimum_peak",
        "s09_minimum_excursion",
        "filtered_classification",
        "filtered_exact_status",
        "filtered_minimum_peak",
        "filtered_minimum_excursion",
        "intervention_created_impossibility",
        "preexisting_exact_impossibility",
        "graph_suppressed_edge_count",
    ]
    joined = pairs.merge(
        reachability[columns], on=keys, how="left", validate="many_to_one"
    )
    if len(joined) != len(pairs) or joined.filtered_exact_status.isna().any():
        raise RuntimeError("hard stop: incomplete exact reachability pair join")
    joined["completion_censored"] = (
        joined.filtered_stop_reason.eq("event_budget")
        & joined.filtered_exact_status.eq("reachable")
    )
    joined["exact_impossibility_not_censoring"] = (
        joined.filtered_stop_reason.eq("event_budget")
        & ~joined.filtered_exact_status.eq("reachable")
    )
    joined["pair_outcome_classification"] = joined.apply(
        _pair_classification, axis=1
    )
    return joined


def _bootstrap_mean(
    frame: pd.DataFrame, column: str, seed_key: str
) -> tuple[float | None, float | None, float | None]:
    usable = frame[["source_family_ordinal", "source_state_ordinal", column]].dropna()
    if usable.empty:
        return None, None, None
    clusters = (
        usable.groupby(["source_family_ordinal", "source_state_ordinal"])[column]
        .mean()
        .to_numpy(dtype=float)
    )
    mean = float(clusters.mean())
    if len(clusters) == 1:
        return mean, mean, mean
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clusters), size=(BOOTSTRAPS, len(clusters)))
    values = clusters[indices].mean(axis=1)
    return mean, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_effects(pairs: pd.DataFrame) -> pd.DataFrame:
    primary = pairs[pairs.filter_threshold == 0].copy()
    primary["delta_selected_final"] = [
        row[f"delta_final_{row.filter_metric}"] for _, row in primary.iterrows()
    ]
    strata: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("intervention", ["intervention_type"]),
        ("s09_exact", ["s09_exact_start_status"]),
        ("s09_observed", ["s09_observed_status"]),
        ("filtered_exact", ["filtered_exact_status"]),
        (
            "exact_by_observed",
            ["s09_exact_start_status", "s09_observed_status", "filtered_exact_status"],
        ),
        (
            "intervention_by_exact",
            ["intervention_type", "s09_exact_start_status", "filtered_exact_status"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for metric, metric_frame in primary.groupby("filter_metric", sort=True):
        for stratum_type, columns in strata:
            groups: Iterable[tuple[Any, pd.DataFrame]]
            if columns:
                groups = metric_frame.groupby(columns, dropna=False, sort=True)
            else:
                groups = [((), metric_frame)]
            for key, group in groups:
                keys = key if isinstance(key, tuple) else (key,)
                row: dict[str, Any] = {
                    "filter_metric": metric,
                    "filter_threshold": 0,
                    "stratum_type": stratum_type,
                    "stratum": (
                        "overall"
                        if not columns
                        else "|".join(f"{name}={value}" for name, value in zip(columns, keys))
                    ),
                    "pair_count": len(group),
                    "source_start_count": len(
                        group[["source_family_ordinal", "source_state_ordinal"]].drop_duplicates()
                    ),
                    "control_completion_fraction": float(group.control_completed.mean()),
                    "filtered_completion_fraction": float(group.filtered_completed.mean()),
                    "created_impossibility_fraction": float(
                        group.intervention_created_impossibility.mean()
                    ),
                    "reachable_censoring_fraction": float(group.completion_censored.mean()),
                    "efficiency_comparable_count": int(group.efficiency_comparable.sum()),
                    "suppressed_total": int(group.suppressed_count.sum()),
                }
                for column in (
                    "delta_completed",
                    "delta_selected_final",
                    "delta_spent_activations",
                    "efficiency_delta_activations",
                    "efficiency_delta_full_ledger_unit_cost",
                ):
                    mean, low, high = _bootstrap_mean(
                        group, column, f"S10/{metric}/{stratum_type}/{row['stratum']}/{column}"
                    )
                    row[f"mean_{column}"] = mean
                    row[f"ci_low_{column}"] = low
                    row[f"ci_high_{column}"] = high
                rows.append(row)

    replicate_zero = pairs[pairs.replicate_index == 0].copy()
    threshold_rows: list[dict[str, Any]] = []
    for (metric, threshold), group in replicate_zero.groupby(
        ["filter_metric", "filter_threshold"], sort=True
    ):
        threshold_rows.append(
            {
                "filter_metric": metric,
                "filter_threshold": int(threshold),
                "stratum_type": "threshold_sensitivity_replicate0",
                "stratum": "all_structural_contexts_replicate0",
                "pair_count": len(group),
                "source_start_count": len(
                    group[["source_family_ordinal", "source_state_ordinal"]].drop_duplicates()
                ),
                "control_completion_fraction": float(group.control_completed.mean()),
                "filtered_completion_fraction": float(group.filtered_completed.mean()),
                "created_impossibility_fraction": float(
                    group.intervention_created_impossibility.mean()
                ),
                "reachable_censoring_fraction": float(group.completion_censored.mean()),
                "efficiency_comparable_count": int(group.efficiency_comparable.sum()),
                "suppressed_total": int(group.suppressed_count.sum()),
            }
        )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(threshold_rows)], ignore_index=True)


def main() -> None:
    started = time.perf_counter()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("hard stop: nonempty S10 artifact directory already exists")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    before = {str(path): sha256_file(path) for path in input_paths()}
    write_json(CACHE / "input_hashes_before.json", before)

    _, families = load_families()
    controls = load_controls()
    reachability = build_filtered_reachability(controls)
    write_parquet(
        OUTPUT / "filtered_structural_reachability.parquet",
        reachability,
        "e03.s10.filtered_structural_reachability.v1",
    )

    grouped_tasks: list[tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]] = []
    for _, group in controls.groupby(
        ["source_family_ordinal", "source_state_ordinal"], sort=True
    ):
        records = group.to_dict("records")
        payloads = {
            int(ordinal): families[int(ordinal)].canonical_dict()
            for ordinal in group.arm_family_ordinal.unique()
        }
        grouped_tasks.append((records, payloads))
    if len(grouped_tasks) != 407:
        raise RuntimeError("hard stop: execution task count is not 407")

    result_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for task_results, task_audits, task_pairs in executor.map(
            run_source_task, grouped_tasks, chunksize=1
        ):
            result_rows.extend(task_results)
            audit_rows.extend(task_audits)
            pair_rows.extend(task_pairs)

    results = pd.DataFrame(result_rows)
    audits = pd.DataFrame(audit_rows)
    pairs = pd.DataFrame(pair_rows)
    if len(results) != 25_224 + 100_896 + 13_312:
        raise RuntimeError(f"hard stop: result count mismatch {len(results)}")
    if len(audits) != 114_208 or len(pairs) != 114_208:
        raise RuntimeError("hard stop: filter/pair count mismatch")
    if not (
        audits.event_partition_valid.all()
        and audits.eligible_partition_valid.all()
        and audits.filter_decision_exact.all()
        and audits.false_positive_count.eq(0).all()
        and audits.false_negative_count.eq(0).all()
        and audits.audited_event_count.eq(audits.trace_event_count).all()
    ):
        raise RuntimeError("hard stop: filter correctness audit failed")
    pair_boolean_fields = [
        "pre_dynamic_state_equal",
        "scenario_id_equal",
        "actor_token_common_prefix_equal",
        "side_token_common_prefix_equal",
        "side_consumption_common_prefix_equal",
        "first_divergence_equals_first_suppression",
        "pre_divergence_identity",
    ]
    if not all(pairs[field].fillna(False).all() for field in pair_boolean_fields):
        raise RuntimeError("hard stop: pair identity/common-stream audit failed")
    divergent = pairs.suppressed_count > 0
    if not (
        pairs.loc[divergent, "divergence_proposal_equal"].all()
        and pairs.loc[divergent, "divergence_prestate_equal"].all()
        and pairs.loc[divergent, "divergence_decision_is_filter_only"].all()
        and pairs.loc[~divergent, "no_suppression_full_trace_equal"].all()
    ):
        raise RuntimeError("hard stop: first-divergence isolation failed")

    pairs = join_structural_labels(pairs, reachability)
    summaries = summarize_effects(pairs)
    write_parquet(
        OUTPUT / "action_suppression_results.parquet",
        results.sort_values(["condition", "run_id"]),
        "e03.s10.action_suppression_result.v1",
    )
    write_parquet(
        OUTPUT / "filter_audit.parquet",
        audits.sort_values("run_id"),
        "e03.s10.filter_audit.v1",
    )
    write_parquet(
        OUTPUT / "paired_effects.parquet",
        pairs.sort_values("pair_id"),
        "e03.s10.paired_effect.v1",
    )
    write_parquet(
        OUTPUT / "effect_summary.parquet",
        summaries,
        "e03.s10.effect_summary.v1",
    )
    environment = {
        "schemaVersion": "e03.s10.environment.v1",
        "researchStepId": "S10",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workerCount": WORKERS,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    write_json(OUTPUT / "environment.json", environment)
    summary = {
        "schemaVersion": "e03.s10.build_summary.v1",
        "researchStepId": "S10",
        "controls": 25_224,
        "primaryPairs": 100_896,
        "thresholdSensitivityPairs": 13_312,
        "resultRows": len(results),
        "filterAuditRows": len(audits),
        "filteredReachabilityRows": len(reachability),
        "sourceStarts": 407,
        "structuralContexts": 3_328,
        "armFamilies": 394,
        "workerCount": WORKERS,
        "elapsedSeconds": time.perf_counter() - started,
    }
    write_json(OUTPUT / "build_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
