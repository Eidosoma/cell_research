#!/usr/bin/env python3
"""Build the exhaustive E03 S09 matched barrier-intervention corpus."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from reference_simulator.model import Architecture, FaultMode, canonical_json_bytes
from src.detours.barrier_interventions import (
    METRICS,
    compare_to_baseline,
    conditioned_seed,
    execute_intervention_arm,
    intervention_families,
)
from src.detours.state_space import FamilySpec


OUTPUT = Path("/artifacts/research_steps/S09")
S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
S06 = Path("/artifacts/research_steps/S06")
S07 = Path("/artifacts/research_steps/S07")
S08 = Path("/artifacts/research_steps/S08")
REPOSITORY = Path(__file__).resolve().parents[1]
MAX_ACTIVATIONS = 2048
CELL_VIEW_REPLICATES = 8
TRADITIONAL_REPLICATES = 1
BOOTSTRAPS = 10_000
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
CLASS_LABELS = {
    0: "complete_start",
    1: "reachable_no_detour",
    2: "necessary_detour",
    3: "unreachable_active",
    4: "quiescent",
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
        raise ValueError(f"refusing to write empty required table {path.name}")
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema_version.encode(), b"researchStepId": b"S09"}
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
        S04 / "canonical_state_encoder.md",
        S05 / "research_step_full_results.md",
        S05 / "graph_corpus_manifest.json",
        S06 / "research_step_full_results.md",
        S06 / "necessary_detour_spec.md",
        S07 / "research_step_full_results.md",
        S07 / "path_solutions.parquet",
        S07 / "necessity_prevalence_by_family.parquet",
        S08 / "research_step_full_results.md",
        S08 / "behavior_necessity_comparison.parquet",
        REPOSITORY / "analysis/s09_barrier_intervention_contract.json",
        REPOSITORY / "src/detours/barrier_interventions.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
    ]
    paths.extend(sorted(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md")))
    return paths


def family_context() -> tuple[pd.DataFrame, dict[int, FamilySpec], dict[str, int]]:
    inventory = pq.read_table(S04 / "state_family_inventory.parquet").to_pandas()
    by_ordinal: dict[int, FamilySpec] = {}
    by_id: dict[str, int] = {}
    for row in inventory.itertuples(index=False):
        family = FamilySpec.from_canonical_dict(json.loads(row.canonical_family_json))
        ordinal = int(row.family_ordinal)
        by_ordinal[ordinal] = family
        by_id[family.family_id] = ordinal
    return inventory, by_ordinal, by_id


def load_source_labels(
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_families = inventory[
        inventory.n.isin([4, 5])
        & (inventory.fault_mode == "stuck")
        & (inventory.fault_count == 1)
    ]
    family_ordinals = [int(value) for value in source_families.family_ordinal]
    dataset = ds.dataset(S07 / "path_solutions.parquet", format="parquet")
    necessary = dataset.to_table(
        filter=ds.field("family_ordinal").isin(family_ordinals)
        & (ds.field("classification_code") == 2)
    ).to_pandas()
    starts = (
        necessary[["family_ordinal", "state_ordinal"]]
        .drop_duplicates()
        .sort_values(["family_ordinal", "state_ordinal"])
        .reset_index(drop=True)
    )
    if len(starts) != 407:
        raise RuntimeError(f"hard stop: expected 407 necessary starts, found {len(starts)}")
    counts = starts.merge(
        inventory[["family_ordinal", "n", "architecture"]], on="family_ordinal"
    )
    if counts.groupby("n").size().to_dict() != {4: 371, 5: 36}:
        raise RuntimeError(f"hard stop: n=4/5 coverage drifted: {counts.groupby('n').size().to_dict()}")

    retained_ordinals = sorted(int(value) for value in starts.family_ordinal.unique())
    all_rows = dataset.to_table(
        filter=ds.field("family_ordinal").isin(retained_ordinals)
    ).to_pandas()
    labels = starts.merge(all_rows, on=["family_ordinal", "state_ordinal"], how="left")
    if len(labels) != len(starts) * 4:
        raise RuntimeError("hard stop: incomplete four-metric S07 source labels")
    labels["metric"] = labels.metric_code.map({value: key for key, value in METRIC_CODES.items()})
    labels["classification"] = labels.classification_code.map(CLASS_LABELS)
    return starts, labels


def arm_task(
    source_family_ordinal: int,
    state_ordinal: int,
    source_family: FamilySpec,
    family_id_to_ordinal: Mapping[str, int],
    replicate_index: int,
) -> dict[str, Any]:
    focal = next(
        index for index, fault in enumerate(source_family.faults) if fault == FaultMode.STUCK
    )
    coupling_key = f"E03/S09/pair/v1:{source_family_ordinal}:{state_ordinal}"
    seed, attempts = conditioned_seed(
        coupling_key, focal, source_family.n, replicate_index
    ) if source_family.architecture == Architecture.CELL_VIEW else (0, 0)
    arms: list[dict[str, Any]] = []
    for intervention, variant, target, family in intervention_families(source_family, focal):
        arm_ordinal = family_id_to_ordinal.get(family.family_id)
        if arm_ordinal is None:
            raise RuntimeError(
                f"hard stop: intervention family absent from S04: {source_family_ordinal} {variant}"
            )
        arms.append(
            {
                "intervention": intervention,
                "variant": variant,
                "target": target,
                "arm_family_ordinal": arm_ordinal,
                "family": family.canonical_dict(),
            }
        )
    return {
        "source_family_ordinal": source_family_ordinal,
        "state_ordinal": state_ordinal,
        "focal": focal,
        "replicate_index": replicate_index,
        "coupling_key": coupling_key,
        "seed": seed,
        "attempts": attempts,
        "arms": arms,
    }


def run_block(task: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, Any]]] = {}
    baseline_row: dict[str, Any] | None = None
    for arm in task["arms"]:
        family = FamilySpec.from_canonical_dict(arm["family"])
        row, trace = execute_intervention_arm(
            family,
            int(task["state_ordinal"]),
            source_family_ordinal=int(task["source_family_ordinal"]),
            arm_family_ordinal=int(arm["arm_family_ordinal"]),
            intervention_type=str(arm["intervention"]),
            arm_variant=str(arm["variant"]),
            focal_index=int(task["focal"]),
            target_index=(int(arm["target"]) if arm["target"] is not None else None),
            replicate_index=int(task["replicate_index"]),
            coupling_key=str(task["coupling_key"]),
            coupling_seed=int(task["seed"]),
            seed_search_attempts=int(task["attempts"]),
            max_activations=MAX_ACTIVATIONS,
        )
        rows.append(row)
        traces[row["run_id"]] = trace
        if row["intervention_type"] == "baseline":
            baseline_row = row
    if baseline_row is None:
        raise AssertionError("matched block has no baseline")
    pairs = [
        compare_to_baseline(
            baseline_row,
            traces[baseline_row["run_id"]],
            row,
            traces[row["run_id"]],
        )
        for row in rows
        if row["intervention_type"] != "baseline"
    ]
    return rows, pairs


def load_arm_optima(
    results: pd.DataFrame,
    source_labels: pd.DataFrame,
) -> pd.DataFrame:
    persistent = results[results.intervention_type != "activate"][
        [
            "source_family_ordinal", "source_state_ordinal", "arm_family_ordinal",
            "arm_state_ordinal", "intervention_type", "arm_variant", "n",
            "architecture", "direction", "policy_profile", "focal_barrier_id",
            "target_cell_id",
        ]
    ].drop_duplicates()
    arm_families = sorted(int(value) for value in persistent.arm_family_ordinal.unique())
    dataset = ds.dataset(S07 / "path_solutions.parquet", format="parquet")
    arm_paths = dataset.to_table(
        filter=ds.field("family_ordinal").isin(arm_families)
    ).to_pandas()
    joined = persistent.merge(
        arm_paths,
        left_on=["arm_family_ordinal", "arm_state_ordinal"],
        right_on=["family_ordinal", "state_ordinal"],
        how="left",
        validate="many_to_many",
    )
    if len(joined) != len(persistent) * 4:
        raise RuntimeError("hard stop: persistent intervention S07 join is incomplete")
    joined["metric"] = joined.metric_code.map({value: key for key, value in METRIC_CODES.items()})
    joined["arm_classification"] = joined.classification_code.map(CLASS_LABELS)
    baseline = source_labels[
        [
            "family_ordinal", "state_ordinal", "metric_code", "classification",
            "minimum_peak", "minimum_excursion", "metric_level",
        ]
    ].rename(
        columns={
            "family_ordinal": "source_family_ordinal",
            "state_ordinal": "source_state_ordinal",
            "classification": "baseline_classification",
            "minimum_peak": "baseline_minimum_peak",
            "minimum_excursion": "baseline_minimum_excursion",
            "metric_level": "baseline_metric_level",
        }
    )
    joined = joined.merge(
        baseline,
        on=["source_family_ordinal", "source_state_ordinal", "metric_code"],
        how="left",
        validate="many_to_one",
    )
    joined = joined.rename(
        columns={
            "minimum_peak": "arm_minimum_peak",
            "minimum_excursion": "arm_minimum_excursion",
            "metric_level": "arm_metric_level",
        }
    )
    joined["arm_goal_reachable"] = joined.arm_classification.isin(
        ["complete_start", "reachable_no_detour", "necessary_detour"]
    )
    joined["delta_minimum_excursion"] = np.where(
        joined.arm_goal_reachable,
        joined.arm_minimum_excursion - joined.baseline_minimum_excursion,
        np.nan,
    )
    joined["necessity_resolved_reachable_no_detour"] = (
        (joined.baseline_classification == "necessary_detour")
        & (joined.arm_classification == "reachable_no_detour")
    )
    joined["necessary_detour_persisted"] = (
        (joined.baseline_classification == "necessary_detour")
        & (joined.arm_classification == "necessary_detour")
    )
    joined["reachability_destroyed"] = (
        (joined.baseline_classification == "necessary_detour")
        & joined.arm_classification.isin(["unreachable_active", "quiescent"])
    )
    joined["necessity_created"] = (
        (joined.baseline_classification != "necessary_detour")
        & (joined.arm_classification == "necessary_detour")
    )
    columns = [
        "source_family_ordinal", "source_state_ordinal", "arm_family_ordinal",
        "arm_state_ordinal", "intervention_type", "arm_variant", "n",
        "architecture", "direction", "policy_profile", "focal_barrier_id",
        "target_cell_id", "metric", "metric_code", "baseline_classification",
        "arm_classification", "baseline_metric_level", "arm_metric_level",
        "baseline_minimum_peak", "arm_minimum_peak", "baseline_minimum_excursion",
        "arm_minimum_excursion", "arm_goal_reachable", "delta_minimum_excursion",
        "necessity_resolved_reachable_no_detour", "necessary_detour_persisted",
        "reachability_destroyed", "necessity_created",
    ]
    return joined[columns].sort_values(
        ["source_family_ordinal", "source_state_ordinal", "arm_variant", "metric_code"]
    ).reset_index(drop=True)


def cluster_interval(values: pd.Series, key: str) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    median = float(np.median(array))
    if len(array) == 1:
        return mean, median, mean, mean
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    boot = array[samples].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean, median, float(low), float(high)


def effect_summaries(
    pairs: pd.DataFrame,
    metric_effects: pd.DataFrame,
    optimum: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_groups(
        frame: pd.DataFrame,
        *,
        analysis_domain: str,
        metric: str,
        outcome: str,
        value_column: str,
        scope: str,
    ) -> None:
        groups = ["n", "architecture", "intervention_type"]
        for keys, group in frame.groupby(groups, sort=True):
            per_start = group.groupby(
                ["source_family_ordinal", "source_state_ordinal"], sort=True
            )[value_column].mean()
            mean, median, low, high = cluster_interval(
                per_start,
                f"{analysis_domain}:{metric}:{outcome}:{scope}:{keys}",
            )
            rows.append(
                {
                    "analysis_domain": analysis_domain,
                    "metric": metric,
                    "outcome": outcome,
                    "scope": scope,
                    "n": int(keys[0]),
                    "architecture": keys[1],
                    "intervention_type": keys[2],
                    "structural_start_count": len(per_start),
                    "arm_comparison_count": len(group),
                    "mean": mean,
                    "median": median,
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "positive_start_fraction": float((per_start > 0).mean()),
                    "negative_start_fraction": float((per_start < 0).mean()),
                    "zero_start_fraction": float((per_start == 0).mean()),
                    "bootstrap_resamples": BOOTSTRAPS,
                }
            )

    for outcome, column in (
        ("completion", "delta_completed"),
        ("activations", "delta_activations"),
        ("accepted_swaps", "delta_accepted_swaps"),
        ("full_ledger_unit_cost", "delta_full_ledger_unit_cost"),
    ):
        add_groups(
            pairs,
            analysis_domain="observed_paired",
            metric="__all__",
            outcome=outcome,
            value_column=column,
            scope="all_source_starts",
        )
    for metric in METRICS:
        subset = metric_effects[metric_effects.metric == metric]
        for scope, scoped in (
            ("all_source_starts", subset),
            ("source_metric_necessary", subset[subset.source_metric_necessary]),
        ):
            if scoped.empty:
                continue
            for outcome, column in (
                ("peak_distance", "delta_peak"),
                ("excursion", "delta_excursion"),
                ("final_distance", "delta_final"),
            ):
                add_groups(
                    scoped,
                    analysis_domain="observed_paired",
                    metric=metric,
                    outcome=outcome,
                    value_column=column,
                    scope=scope,
                )
    necessary_optimum = optimum[optimum.baseline_classification == "necessary_detour"]
    for metric in METRICS:
        metric_optimum = necessary_optimum[necessary_optimum.metric == metric]
        if metric_optimum.empty:
            continue
        reachable_optimum = metric_optimum[metric_optimum.arm_goal_reachable]
        if not reachable_optimum.empty:
            add_groups(
                reachable_optimum,
                analysis_domain="exact_structural_optimum",
                metric=metric,
                outcome="minimum_excursion",
                value_column="delta_minimum_excursion",
                scope="source_metric_necessary_arm_reachable",
            )
        for outcome, column in (
            ("necessity_resolved_reachable_no_detour", "necessity_resolved_reachable_no_detour"),
            ("necessary_detour_persisted", "necessary_detour_persisted"),
            ("reachability_destroyed", "reachability_destroyed"),
        ):
            add_groups(
                metric_optimum,
                analysis_domain="exact_structural_optimum",
                metric=metric,
                outcome=outcome,
                value_column=column,
                scope="source_metric_necessary",
            )
    return pd.DataFrame(rows)


def representative_pairs(metric_effects: pd.DataFrame) -> pd.DataFrame:
    candidates = metric_effects[metric_effects.source_metric_necessary].copy()
    candidates["rank_abs_excursion"] = candidates.delta_excursion.abs()
    candidates["rank_abs_completion"] = candidates.delta_completed.abs()
    candidates["rank_abs_cost"] = candidates.delta_full_ledger_unit_cost.abs()
    selected = (
        candidates.sort_values(
            [
                "n", "architecture", "intervention_type", "rank_abs_excursion",
                "rank_abs_completion", "rank_abs_cost", "arm_run_id",
            ],
            ascending=[True, True, True, False, False, False, True],
        )
        .groupby(["n", "architecture", "intervention_type"], sort=True)
        .head(1)
        .reset_index(drop=True)
    )
    return selected


def replay_trace_rows(
    selected: pd.DataFrame,
    results: pd.DataFrame,
    families: Mapping[int, FamilySpec],
) -> list[dict[str, Any]]:
    by_run = results.set_index("run_id")
    rows: list[dict[str, Any]] = []
    for panel_index, selected_row in selected.iterrows():
        panel_id = f"panel-{panel_index:02d}"
        for role, run_id in (
            ("baseline", selected_row.baseline_run_id),
            ("intervention", selected_row.arm_run_id),
        ):
            source = by_run.loc[run_id]
            family = families[int(source.arm_family_ordinal)]
            replay, trace = execute_intervention_arm(
                family,
                int(source.source_state_ordinal),
                source_family_ordinal=int(source.source_family_ordinal),
                arm_family_ordinal=int(source.arm_family_ordinal),
                intervention_type=str(source.intervention_type),
                arm_variant=str(source.arm_variant),
                focal_index=int(str(source.focal_barrier_id)[1:]),
                target_index=(
                    int(str(source.target_cell_id)[1:])
                    if pd.notna(source.target_cell_id)
                    else None
                ),
                replicate_index=int(source.replicate_index),
                coupling_key=str(source.coupling_key),
                coupling_seed=int(source.coupling_seed),
                seed_search_attempts=int(source.seed_search_attempts),
                max_activations=int(source.event_budget),
            )
            if replay["compact_trace_sha256"] != source.compact_trace_sha256:
                raise RuntimeError("representative trace replay mismatch")
            initial = {
                "panel_id": panel_id,
                "path_role": role,
                "run_id": run_id,
                "intervention_type": selected_row.intervention_type,
                "metric": selected_row.metric,
                "event_index": -1,
                "actor_id": None,
                "decision": "initial",
                "occupancy": None,
                "terminal": source.initial_terminal,
            }
            for metric in METRICS:
                initial[metric] = int(source[f"start_{metric}"])
            rows.append(initial)
            for event in trace:
                item = {
                    "panel_id": panel_id,
                    "path_role": role,
                    "run_id": run_id,
                    "intervention_type": selected_row.intervention_type,
                    "metric": selected_row.metric,
                    "event_index": int(event["event_index"]),
                    "actor_id": event["actor_id"],
                    "decision": event["decision"],
                    "occupancy": json.dumps(event["after_occupancy"], separators=(",", ":")),
                    "terminal": event["terminal"],
                }
                item.update({name: int(event["after_levels"][name]) for name in METRICS})
                rows.append(item)
    return rows


def plot_effects(summary: pd.DataFrame, trace_rows: pd.DataFrame) -> None:
    effects = summary[
        (summary.analysis_domain == "observed_paired")
        & (summary.outcome == "excursion")
        & (summary.scope == "source_metric_necessary")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    if not effects.empty:
        effects["label"] = (
            "n=" + effects.n.astype(str) + " " + effects.architecture.str.replace("_", " ")
            + "\n" + effects.metric.str.replace("_", " ")
        )
        interventions = ["remove", "activate", "move", "add"]
        labels = list(dict.fromkeys(effects.label))
        matrix = np.full((len(interventions), len(labels)), np.nan)
        for row in effects.itertuples(index=False):
            matrix[interventions.index(row.intervention_type), labels.index(row.label)] = row.mean
        image = axes[0].imshow(matrix, cmap="coolwarm", vmin=-max(1, np.nanmax(abs(matrix))), vmax=max(1, np.nanmax(abs(matrix))))
        axes[0].set_yticks(range(len(interventions)), interventions)
        axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axes[0].set_title("Mean paired change in necessary-metric excursion")
        fig.colorbar(image, ax=axes[0], label="intervention minus baseline")
    panels = sorted(trace_rows.panel_id.unique())[:4]
    for panel in panels:
        part = trace_rows[trace_rows.panel_id == panel]
        metric = str(part.metric.iloc[0])
        for role, line in part.groupby("path_role"):
            axes[1].plot(line.event_index + 1, line[metric], alpha=0.75, label=f"{panel} {role}")
    axes[1].set_title("Representative matched necessary-start traces")
    axes[1].set_xlabel("charged opportunity (initial = 0)")
    axes[1].set_ylabel("named distance")
    if panels:
        axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "barrier_effects.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    before = {str(path): sha256_file(path) for path in input_paths()}
    inventory, families, family_id_to_ordinal = family_context()
    starts, source_labels = load_source_labels(inventory)
    source_meta = starts.merge(
        inventory[
            [
                "family_ordinal", "n", "architecture", "direction", "policy_profile",
                "fault_count", "fault_identity_ids",
            ]
        ],
        on="family_ordinal",
        how="left",
    )
    necessary_signatures = (
        source_labels[source_labels.classification_code == 2]
        .groupby(["family_ordinal", "state_ordinal"])
        .metric.apply(lambda values: "+".join(sorted(values)))
        .rename("source_necessary_signature")
        .reset_index()
    )
    source_meta = source_meta.merge(
        necessary_signatures, on=["family_ordinal", "state_ordinal"], how="left"
    )
    source_meta["activation_intervention_defined"] = source_meta.architecture == "cell_view"
    coverage = source_labels.merge(
        source_meta[
            [
                "family_ordinal", "state_ordinal", "n", "architecture", "direction",
                "policy_profile", "fault_identity_ids", "source_necessary_signature",
                "activation_intervention_defined",
            ]
        ],
        on=["family_ordinal", "state_ordinal"],
        how="left",
    )
    coverage["selected_for_s09"] = True
    coverage["source_goal_reachable"] = coverage.classification.isin(
        ["complete_start", "reachable_no_detour", "necessary_detour"]
    )

    tasks: list[dict[str, Any]] = []
    for row in starts.itertuples(index=False):
        family = families[int(row.family_ordinal)]
        replicates = (
            CELL_VIEW_REPLICATES
            if family.architecture == Architecture.CELL_VIEW
            else TRADITIONAL_REPLICATES
        )
        for replicate in range(replicates):
            tasks.append(
                arm_task(
                    int(row.family_ordinal), int(row.state_ordinal), family,
                    family_id_to_ordinal, replicate,
                )
            )
    workers = min(8, os.cpu_count() or 1)
    result_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for runs, pairs in pool.map(run_block, tasks, chunksize=4):
            result_rows.extend(runs)
            pair_rows.extend(pairs)
    results = pd.DataFrame(result_rows)
    pairs = pd.DataFrame(pair_rows)
    if not results.trace_complete.all() or not results.ledger_identities_valid.all():
        raise RuntimeError("hard stop: trace completeness or ledger identity failure")
    if not pairs.pre_dynamic_state_equal.all():
        raise RuntimeError("hard stop: cross-arm pre-dynamic-state mismatch")
    cell_pairs = pairs[pairs.architecture == "cell_view"]
    if not cell_pairs.actor_token_common_prefix_equal.all() or not cell_pairs.side_token_common_prefix_equal.all():
        raise RuntimeError("hard stop: common scheduler token mismatch")
    pulse_rows = results[results.intervention_type == "activate"]
    if len(pulse_rows) == 0 or not pulse_rows.pulse_delivered.fillna(False).all():
        raise RuntimeError("hard stop: a cell-view activation pulse was not delivered")

    results = results.merge(
        source_meta[
            [
                "family_ordinal", "state_ordinal", "source_necessary_signature"
            ]
        ],
        left_on=["source_family_ordinal", "source_state_ordinal"],
        right_on=["family_ordinal", "state_ordinal"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["family_ordinal", "state_ordinal"])
    pairs = pairs.merge(
        source_meta[
            ["family_ordinal", "state_ordinal", "source_necessary_signature"]
        ],
        left_on=["source_family_ordinal", "source_state_ordinal"],
        right_on=["family_ordinal", "state_ordinal"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["family_ordinal", "state_ordinal"])

    optimum = load_arm_optima(results, source_labels)
    source_label_lookup = source_labels.set_index(
        ["family_ordinal", "state_ordinal", "metric"]
    )
    metric_rows: list[dict[str, Any]] = []
    for row in pairs.itertuples(index=False):
        for metric in METRICS:
            label = source_label_lookup.loc[
                (int(row.source_family_ordinal), int(row.source_state_ordinal), metric)
            ]
            metric_rows.append(
                {
                    "source_family_ordinal": int(row.source_family_ordinal),
                    "source_state_ordinal": int(row.source_state_ordinal),
                    "pair_block_id": row.pair_block_id,
                    "baseline_run_id": row.baseline_run_id,
                    "arm_run_id": row.arm_run_id,
                    "n": int(row.n),
                    "architecture": row.architecture,
                    "direction": row.direction,
                    "policy_profile": row.policy_profile,
                    "intervention_type": row.intervention_type,
                    "arm_variant": row.arm_variant,
                    "target_cell_id": row.target_cell_id,
                    "metric": metric,
                    "source_metric_classification": label.classification,
                    "source_metric_necessary": label.classification == "necessary_detour",
                    "source_minimum_excursion": int(label.minimum_excursion),
                    "delta_peak": int(getattr(row, f"delta_peak_{metric}")),
                    "delta_excursion": int(getattr(row, f"delta_excursion_{metric}")),
                    "delta_final": int(getattr(row, f"delta_final_{metric}")),
                    "delta_completed": int(row.delta_completed),
                    "delta_activations": int(row.delta_activations),
                    "delta_full_ledger_unit_cost": int(row.delta_full_ledger_unit_cost),
                }
            )
    metric_effects = pd.DataFrame(metric_rows)
    summary = effect_summaries(pairs, metric_effects, optimum)

    isolation_rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(
        ["source_family_ordinal", "source_state_ordinal", "arm_variant"], sort=True
    ):
        source_family = families[int(keys[0])]
        arm_family = families[int(group.arm_family_ordinal.iloc[0])]
        diffs = [
            index for index, (left, right) in enumerate(zip(source_family.faults, arm_family.faults))
            if left != right
        ]
        intervention = group.intervention_type.iloc[0]
        expected_diff_count = {"baseline": 0, "activate": 0, "remove": 1, "move": 2, "add": 1}[intervention]
        isolation_rows.append(
            {
                "source_family_ordinal": int(keys[0]),
                "source_state_ordinal": int(keys[1]),
                "arm_variant": keys[2],
                "intervention_type": intervention,
                "arm_family_ordinal": int(group.arm_family_ordinal.iloc[0]),
                "replicate_count": len(group),
                "pre_dynamic_state_hash_count": int(group.pre_dynamic_state_sha256.nunique()),
                "fault_difference_indices": json.dumps(diffs),
                "fault_difference_count": len(diffs),
                "expected_fault_difference_count": expected_diff_count,
                "policies_equal": source_family.policies == arm_family.policies,
                "direction_equal": source_family.direction == arm_family.direction,
                "architecture_equal": source_family.architecture == arm_family.architecture,
                "isolated": len(diffs) == expected_diff_count
                and source_family.policies == arm_family.policies
                and source_family.direction == arm_family.direction
                and source_family.architecture == arm_family.architecture,
            }
        )
    isolation = pd.DataFrame(isolation_rows)
    if not isolation.isolated.all():
        raise RuntimeError("hard stop: intervention isolation audit failed")

    selected = representative_pairs(metric_effects)
    trace_rows = pd.DataFrame(replay_trace_rows(selected, results, families))
    plot_effects(summary, trace_rows)

    write_parquet(OUTPUT / "barrier_interventions.parquet", results, "e03.s09.barrier_interventions.v1")
    write_parquet(OUTPUT / "paired_effects.parquet", pairs, "e03.s09.paired_effects.v1")
    write_parquet(OUTPUT / "paired_metric_effects.parquet", metric_effects, "e03.s09.paired_metric_effects.v1")
    write_parquet(OUTPUT / "effect_summary.parquet", summary, "e03.s09.effect_summary.v1")
    write_parquet(OUTPUT / "necessary_start_coverage.parquet", coverage, "e03.s09.necessary_start_coverage.v1")
    write_parquet(OUTPUT / "structural_optimum_changes.parquet", optimum, "e03.s09.structural_optimum_changes.v1")
    write_parquet(OUTPUT / "intervention_isolation_audit.parquet", isolation, "e03.s09.intervention_isolation_audit.v1")
    write_parquet(
        OUTPUT / "random_stream_audit.parquet",
        pairs[
            [
                "pair_block_id", "baseline_run_id", "arm_run_id", "n", "architecture",
                "intervention_type", "arm_variant", "common_prefix_event_count",
                "actor_token_common_prefix_equal", "side_token_common_prefix_equal",
                "first_side_consumption_divergence_event",
                "first_decision_divergence_event", "first_dynamic_state_divergence_event",
            ]
        ],
        "e03.s09.random_stream_audit.v1",
    )
    write_parquet(OUTPUT / "paired_traces.parquet", trace_rows, "e03.s09.paired_traces.v1")

    after = {str(path): sha256_file(path) for path in input_paths()}
    immutability = {
        "schemaVersion": "e03.s09.input_immutability.v1",
        "researchStepId": "S09",
        "allUnchanged": before == after,
        "files": [
            {"path": path, "beforeSha256": before[path], "afterSha256": after[path], "unchanged": before[path] == after[path]}
            for path in sorted(before)
        ],
    }
    if not immutability["allUnchanged"]:
        raise RuntimeError("hard stop: an S09 input changed during execution")
    write_json(OUTPUT / "input_immutability.json", immutability)
    environment = {
        "schemaVersion": "e03.s09.environment.v1",
        "researchStepId": "S09",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": workers,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "elapsedSeconds": time.time() - started,
    }
    write_json(OUTPUT / "environment.json", environment)
    write_json(
        OUTPUT / "build_summary.json",
        {
            "schemaVersion": "e03.s09.build_summary.v1",
            "researchStepId": "S09",
            "sourceStarts": len(starts),
            "n4Starts": int((source_meta.n == 4).sum()),
            "n5Starts": int((source_meta.n == 5).sum()),
            "sourceFamilies": int(starts.family_ordinal.nunique()),
            "matchedBlocks": len(tasks),
            "runs": len(results),
            "pairedComparisons": len(pairs),
            "pairedMetricComparisons": len(metric_effects),
            "persistentOptimumRows": len(optimum),
            "representativePanels": int(trace_rows.panel_id.nunique()),
            "eventBudgetCensored": int(results.event_budget_censored.sum()),
            "completions": int(results.completed.sum()),
            "workers": workers,
            "elapsedSeconds": time.time() - started,
        },
    )
    print(json.dumps(json.loads((OUTPUT / "build_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
