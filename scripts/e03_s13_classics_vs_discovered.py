#!/usr/bin/env python3
"""Execute E03 S13: compare classics to discovered DSL policies.

S13 uses completed S08-S12 artifacts to compare Bubble, Insertion, and
Selection regions against evidence-supported discovered policies.  It runs a
bounded same-seed CPU panel on S12-style tasks, joins competence deltas to S10
embeddings, S09 phase-boundary context, S11 class placements, and S12 ablation
context, then stops before S14.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from e02_deterministic_simulator.metrics import aggregation  # noqa: E402
from morphospace import (  # noqa: E402
    CLASSIC_COMPARISON_VERSION,
    CLASSIC_FAMILIES,
    ComparisonTask,
    DSLPolicy,
    NullPolicy,
    PolicyEventSimulator,
    aggregate_s13_vectors,
    build_pairwise_comparison_table,
    build_policy_comparison_universe,
    compute_competence_vector,
    family_summary_table,
    missing_comparison_records,
    pareto_front_table,
    parse_rule_program,
    same_seed_task_delta_table,
    summarize_same_seed_pairs,
    validate_classic_comparison_outputs,
)
from morphospace.competence import canonical_json, delayed_gratification_from_sortedness, sortedness_sign_change_count  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S13"
STEP_NUMBER = 13
TITLE = "Compare classics to discovered rules"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
SEED_COUNT = 3

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = STEP_DIR / "figures"
SELECTED_POLICY_DIR = STEP_DIR / "selected_policy_dsl"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_classics_vs_discovered"

CORPUS_PATH = ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet"
S08_ELITES_PATH = RESULTS_DIR / "e03_qd_elites.parquet"
S09_BOUNDARIES_PATH = RESULTS_DIR / "e03_phase_boundaries.parquet"
S09_RUNS_PATH = RESULTS_DIR / "e03_s09_phase_boundary_runs.parquet"
S10_FEATURES_PATH = RESULTS_DIR / "e03_s10_embedding_features.parquet"
S10_NORMALIZED_PATH = ARTIFACTS_DIR / "research_steps" / "S10" / "normalized_feature_matrix.parquet"
S10_EMBEDDINGS_PATH = RESULTS_DIR / "e03_policy_embeddings.parquet"
S10_NEIGHBORS_PATH = RESULTS_DIR / "e03_s10_embedding_neighbors.parquet"
S11_ASSIGNMENTS_PATH = RESULTS_DIR / "e03_s11_policy_class_assignments.parquet"
S11_PLACEMENTS_PATH = RESULTS_DIR / "e03_s11_classic_null_elite_placements.parquet"
S12_POLICY_SPECS_PATH = RESULTS_DIR / "e03_s12_ablation_policy_specs.parquet"
S12_ABLATIONS_PATH = RESULTS_DIR / "e03_feature_ablations.parquet"
S12_EFFECTS_PATH = RESULTS_DIR / "e03_s12_feature_effect_summary.parquet"

S13_TASKS = (
    ComparisonTask(
        task_id="s13_unique_n5_same_seed",
        task_family="sorting",
        task_panel="s13_classic_discovered_same_seed",
        input_profile="unique_n5_probe",
        initial_values=(5, 1, 4, 2, 3),
        scheduler_seed_base=61000,
        tie_seed_base=71000,
        max_activations=640,
        max_swaps=640,
        max_comparisons=2560,
    ),
    ComparisonTask(
        task_id="s13_duplicate_n5_same_seed",
        task_family="sorting_duplicate_values",
        task_panel="s13_classic_discovered_same_seed",
        input_profile="duplicate_n5_probe",
        initial_values=(2, 3, 1, 2, 1),
        scheduler_seed_base=62000,
        tie_seed_base=72000,
        max_activations=640,
        max_swaps=640,
        max_comparisons=2560,
    ),
    ComparisonTask(
        task_id="s13_stuck_frozen_n5_same_seed",
        task_family="frozen",
        task_panel="s13_classic_discovered_same_seed",
        input_profile="unique_n5_stuck_frozen_probe",
        initial_values=(5, 1, 4, 2, 3),
        scheduler_seed_base=63000,
        tie_seed_base=73000,
        frozen_variant="stuck",
        frozen_positions=(2,),
        max_activations=704,
        max_swaps=704,
        max_comparisons=2816,
    ),
    ComparisonTask(
        task_id="s13_transfer_unique_n6_same_seed",
        task_family="transfer",
        task_panel="s13_classic_discovered_same_seed",
        input_profile="unique_n6_transfer_probe",
        initial_values=(6, 1, 5, 2, 4, 3),
        scheduler_seed_base=64000,
        tie_seed_base=74000,
        max_activations=832,
        max_swaps=832,
        max_comparisons=3328,
    ),
    ComparisonTask(
        task_id="s13_chimera_null_n6_same_seed",
        task_family="chimera",
        task_panel="s13_classic_discovered_same_seed",
        input_profile="alternating_dsl_null_chimera_n6_probe",
        initial_values=(6, 1, 5, 2, 4, 3),
        scheduler_seed_base=65000,
        tie_seed_base=75000,
        chimera_with_null=True,
        candidate_positions=(0, 2, 4),
        max_activations=832,
        max_swaps=832,
        max_comparisons=3328,
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def table_ready(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.columns:
        if prepared[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            prepared[column] = prepared[column].map(
                lambda value: canonical_json(sorted(value) if isinstance(value, set) else value)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
    return prepared


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> list[Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out = table_ready(df)
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append({"path": str(path), "sizeBytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return rows


def run_command(command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 900) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returnCode": int(completed.returncode),
        "success": completed.returncode == 0,
        "runtimeSeconds": time.perf_counter() - started,
        "output": completed.stdout,
    }


def get_git_metadata() -> dict[str, Any]:
    def read(command: list[str]) -> str:
        result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return result.stdout.strip()

    return {
        "branch": read(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": read(["git", "rev-parse", "HEAD"]),
        "statusShort": read(["git", "status", "--short"]),
        "remote": read(["git", "remote", "get-url", "origin"]),
    }


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"success": False, "missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict[str, pd.DataFrame]:
    required_paths = {
        "corpus": CORPUS_PATH,
        "s08_elites": S08_ELITES_PATH,
        "s09_boundaries": S09_BOUNDARIES_PATH,
        "s09_runs": S09_RUNS_PATH,
        "s10_features": S10_FEATURES_PATH,
        "s10_normalized": S10_NORMALIZED_PATH,
        "s10_embeddings": S10_EMBEDDINGS_PATH,
        "s10_neighbors": S10_NEIGHBORS_PATH,
        "s11_assignments": S11_ASSIGNMENTS_PATH,
        "s11_placements": S11_PLACEMENTS_PATH,
        "s12_policy_specs": S12_POLICY_SPECS_PATH,
        "s12_ablations": S12_ABLATIONS_PATH,
        "s12_effects": S12_EFFECTS_PATH,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"S13 requires completed upstream inputs; missing: {missing}")
    for step in ["S08", "S09", "S10", "S11", "S12"]:
        status = load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        if not status.get("success"):
            raise RuntimeError(f"S13 requires successful {step}; got {status}")
    return {key: pd.read_parquet(path) for key, path in required_paths.items()}


def value_counts_conserved(initial_values: tuple[int, ...], final_values: list[int]) -> bool:
    return Counter(map(int, initial_values)) == Counter(map(int, final_values))


def trace_sortedness(result) -> list[float]:
    return [float(row["sortedness_percent"]) for row in result.trace_rows]


def trace_hashes(result) -> list[str]:
    return [str(row["state_hash"]) for row in result.trace_rows]


def make_policies_for_task(program, task: ComparisonTask):
    if not task.chimera_with_null:
        return DSLPolicy(program)
    candidate_positions = set(map(int, task.candidate_positions))
    policies = []
    for position in range(len(task.initial_values)):
        policies.append(DSLPolicy(program) if position in candidate_positions else NullPolicy())
    return policies


def run_one_policy_task_seed(row: pd.Series, task: ComparisonTask, seed_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_id = str(row["policyId"])
    program = parse_rule_program(row["dslProgramJson"])
    scheduler_seed = int(task.scheduler_seed_base + int(seed_index))
    tie_seed = int(task.tie_seed_base + int(seed_index))
    condition_id = f"S13::{task.task_id}::{policy_id}::seed{scheduler_seed}"
    policies = make_policies_for_task(program, task)
    started = time.perf_counter()
    result = PolicyEventSimulator(
        list(task.initial_values),
        policies,
        frozen_positions=task.frozen_positions,
        frozen_variant=task.frozen_variant,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_seed,
        condition_id=condition_id,
        implementation="e03_s13_cpu_same_seed_classic_discovered",
        research_step_id=STEP_ID,
    ).run(
        max_activations=task.max_activations,
        max_swaps=task.max_swaps,
        max_comparisons=task.max_comparisons,
    )
    runtime = time.perf_counter() - started
    sortedness_values = trace_sortedness(result)
    dg = delayed_gratification_from_sortedness(sortedness_values)
    oscillation_proxy = (
        sortedness_sign_change_count(sortedness_values) / max(1, len(sortedness_values) - 1)
        if len(sortedness_values) > 1
        else 0.0
    )
    final_sortedness_score = float(result.final_sortedness_percent / 100.0)
    initial_aggregation = float(aggregation(result.initial_algotypes))
    final_aggregation = float(result.final_aggregation)
    run_record = {
        "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
        "researchStepId": STEP_ID,
        "policyId": policy_id,
        "comparisonRole": str(row["comparisonRole"]),
        "classicFamily": str(row.get("classicFamily", "none")),
        "primaryRole": str(row.get("primaryRole", "")),
        "className": str(row.get("className", "")),
        "classLabel": str(row.get("classLabel", "")),
        "generationMethod": str(row.get("generationMethod", "")),
        "lineageId": str(row.get("lineageId", "")),
        "taskId": task.task_id,
        "taskFamily": task.task_family,
        "taskPanel": task.task_panel,
        "inputProfile": task.input_profile,
        "seedIndex": int(seed_index),
        "schedulerSeed": scheduler_seed,
        "tieBreakerSeed": tie_seed,
        "initialValuesJson": canonical_json(list(task.initial_values)),
        "finalValuesJson": canonical_json(result.final_values),
        "frozenVariant": task.frozen_variant,
        "frozenPositionsJson": canonical_json(list(task.frozen_positions)),
        "chimeraWithNull": bool(task.chimera_with_null),
        "candidatePositionsJson": canonical_json(list(task.candidate_positions)),
        "completed": bool(result.completed),
        "completedNumeric": 1.0 if result.completed else 0.0,
        "stopReason": result.stop_reason,
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalSortednessScore": final_sortedness_score,
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "delayedGratification": float(dg["delayedGratification"]),
        "dgEventCount": int(dg["dgEventCount"]),
        "oscillationProxy": float(oscillation_proxy),
        "initialAggregation": initial_aggregation,
        "finalAggregation": final_aggregation,
        "robustnessScore": None
        if task.frozen_variant == "none"
        else float((1.0 if result.completed else 0.0) * final_sortedness_score),
        "traceStateCount": int(len(result.trace_rows)),
        "traceHashSequenceJson": canonical_json(trace_hashes(result)),
        "runtimeSeconds": float(runtime),
        "valueCountsConserved": bool(value_counts_conserved(task.initial_values, result.final_values)),
    }
    vector = compute_competence_vector(
        result,
        policy_id=policy_id,
        policy_family=str(row.get("family", "unknown")),
        task_id=task.task_id,
        task_family=task.task_family,
        task_panel=task.task_panel,
        input_profile=task.input_profile,
        frozen_variant=task.frozen_variant,
        frozen_count=len(task.frozen_positions),
        replicate_index=seed_index,
        source_metric_source="s13_cpu_same_seed_classic_discovered",
        source_artifact_path=str(STEP_DIR / "same_seed_runs.parquet"),
        transfer_score=final_sortedness_score if task.task_family == "transfer" else None,
        compatibility_score=final_aggregation if task.task_family == "chimera" else None,
    )
    vector.update(
        {
            "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
            "comparisonRole": str(row["comparisonRole"]),
            "classicFamily": str(row.get("classicFamily", "none")),
            "primaryRole": str(row.get("primaryRole", "")),
            "className": str(row.get("className", "")),
            "classLabel": str(row.get("classLabel", "")),
            "generationMethod": str(row.get("generationMethod", "")),
            "lineageId": str(row.get("lineageId", "")),
            "schedulerSeed": scheduler_seed,
            "tieBreakerSeed": tie_seed,
            "backend": "cpu_reference",
        }
    )
    return run_record, vector


def run_same_seed_panel(selected_policies_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    sorted_policies = selected_policies_df.sort_values(["comparisonRole", "classicFamily", "primaryRole", "policyId"], kind="mergesort")
    for _, row in sorted_policies.iterrows():
        for task in S13_TASKS:
            for seed_index in range(SEED_COUNT):
                run_record, vector = run_one_policy_task_seed(row, task, seed_index)
                run_rows.append(run_record)
                vector_rows.append(vector)
    return pd.DataFrame(run_rows), pd.DataFrame(vector_rows)


def write_selected_policy_files(selected_policies_df: pd.DataFrame) -> pd.DataFrame:
    SELECTED_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in selected_policies_df.sort_values(["comparisonRole", "classicFamily", "policyId"], kind="mergesort").iterrows():
        program = parse_rule_program(row["dslProgramJson"])
        policy_id = str(row["policyId"])
        path = SELECTED_POLICY_DIR / f"{policy_id}.json"
        path.write_text(json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
                "policyId": policy_id,
                "comparisonRole": str(row["comparisonRole"]),
                "classicFamily": str(row.get("classicFamily", "none")),
                "primaryRole": str(row.get("primaryRole", "")),
                "dslPath": str(path),
                "dslSha256": sha256_file(path),
                "roundtripSuccess": canonical_json(parse_rule_program(program.to_json()).to_dict()) == canonical_json(program.to_dict()),
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(
    *,
    policy_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    selected = policy_df[policy_df["selectedForS13"]].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    role_colors = {"classic": "#2f5f8f", "discovered": "#8a6b28"}
    for role, group in selected.groupby("comparisonRole"):
        ax.scatter(
            group["sortednessScore"],
            group["energyScore"],
            label=role,
            alpha=0.75,
            s=36 if role == "classic" else 20,
            color=role_colors.get(role, "#777777"),
            edgecolor="black" if role == "classic" else "none",
            linewidth=0.5,
        )
    ax.set_xlabel("broad sortedness score")
    ax.set_ylabel("broad energy score")
    ax.set_title("S13 selected policy competence surface")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES_DIR / "s13_sortedness_energy_scatter.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    if not pairwise_df.empty:
        plot_df = pairwise_df[pairwise_df["sameSeedNetAdvantageScore"].notna()].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        values = [
            pd.to_numeric(plot_df.loc[plot_df["classicFamily"].eq(family), "sameSeedNetAdvantageScore"], errors="coerce").dropna()
            for family in CLASSIC_FAMILIES
        ]
        ax.boxplot(values, tick_labels=CLASSIC_FAMILIES, showfliers=False)
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        ax.set_ylabel("same-seed net advantage, discovered - classic")
        ax.set_title("S13 discovered policy advantage by classic family")
        fig.tight_layout()
        path = FIGURES_DIR / "s13_same_seed_advantage_by_family.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    pca = embeddings_df[embeddings_df["embeddingMethod"].eq("pca") & embeddings_df["embeddingSeed"].eq(0)].copy()
    pca = pca.merge(selected[["policyId", "comparisonRole", "classicFamily"]], on="policyId", how="inner")
    if not pca.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        for role, group in pca.groupby("comparisonRole"):
            ax.scatter(
                group["embeddingDim1"],
                group["embeddingDim2"],
                label=role,
                alpha=0.75,
                s=38 if role == "classic" else 18,
                color=role_colors.get(role, "#777777"),
                edgecolor="black" if role == "classic" else "none",
                linewidth=0.5,
            )
        ax.set_xlabel("PCA dim 1")
        ax.set_ylabel("PCA dim 2")
        ax.set_title("S13 selected policies in S10 PCA embedding")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = FIGURES_DIR / "s13_selected_policy_pca_embedding.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    if not family_summary_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.bar(
            family_summary_df["classicFamily"],
            family_summary_df["discoveredDominatingSameSeedPairCount"],
            color="#687d45",
            label="discovered dominates",
        )
        ax.bar(
            family_summary_df["classicFamily"],
            -family_summary_df["classicDominatingSameSeedPairCount"],
            color="#8f4d3a",
            label="classic dominates",
        )
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        ax.set_ylabel("pair count")
        ax.set_title("S13 same-seed dominance counts")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = FIGURES_DIR / "s13_same_seed_dominance_counts.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def render_comparison_report(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    policy_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    selected = policy_df[policy_df["selectedForS13"]]
    classics = selected[selected["comparisonRole"].eq("classic")]
    discovered = selected[selected["comparisonRole"].eq("discovered")]
    pareto_opt = pareto_df[pareto_df["isParetoOptimal"]] if "isParetoOptimal" in pareto_df.columns else pd.DataFrame()
    top_pairs = pairwise_df.sort_values(
        ["discoveredDominatesClassicSameSeed", "sameSeedNetAdvantageScore"],
        ascending=[False, False],
        kind="mergesort",
    ).head(12)
    rows = [
        "# S13 Classics Vs Discovered Report",
        "",
        "- Research step ID: S13",
        f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
        f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_classics_vs_discovered.parquet'}`, `{RESULTS_DIR / 'e03_s13_policy_comparison_summary.parquet'}`, and `{FIGURES_DIR}/`",
        f"- Validation result: {validation_result}",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "- Lay summary: S13 compares the classic Bubble, Insertion, and Selection policy regions against discovered DSL policies using the same task panels and literal scheduler/tie seeds where S13 reran CPU-reference probes.",
        "",
        "## Design",
        "",
        "- Classics are all S10/S11 policies marked as Bubble, Insertion, or Selection family members.",
        "- Discovered policies are standalone non-null, non-classic DSL records with S07, S09, S12, elite, phase-boundary, or pathological/failure-mode evidence.",
        "- The same-seed panel reuses the S12 task families but assigns identical scheduler and tie-breaker seed values to every selected policy.",
        "- Pairwise comparisons join same-seed deltas to S10 embedding distances, S09 phase-boundary context, S11 class placements, and broad competence/Pareto calls.",
        "",
        "## Counts",
        "",
        f"- Classic policies: {len(classics)} across families `{canonical_json(classics['classicFamily'].value_counts().to_dict())}`",
        f"- Discovered policies: {len(discovered)} roles `{canonical_json(discovered['primaryRole'].value_counts().to_dict())}`",
        f"- Pairwise classic/discovered comparisons: {len(pairwise_df)}",
        f"- Pareto-optimal selected policies: {len(pareto_opt)}",
        "",
        "## Family Summary",
        "",
    ]
    if family_summary_df.empty:
        rows.append("- No family summary rows were available.")
    else:
        rows.extend(
            [
                "| Family | Classics | Discovered Compared | Classic Pareto Optima | Discovered Same-Seed Dominating Pairs | Classic Same-Seed Dominating Pairs | Best Discovered |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in family_summary_df.iterrows():
            rows.append(
                f"| `{row['classicFamily']}` | {int(row['classicPolicyCount'])} | {int(row['discoveredPolicyCountCompared'])} | "
                f"{int(row['classicParetoOptimalCount'])} | {int(row['discoveredDominatingSameSeedPairCount'])} | "
                f"{int(row['classicDominatingSameSeedPairCount'])} | `{row['bestDiscoveredPolicyId']}` |"
            )
    rows.extend(["", "## Top Same-Seed Discovered Advantages", ""])
    if top_pairs.empty:
        rows.append("- No pairwise rows were available.")
    else:
        rows.extend(
            [
                "| Classic Family | Classic Policy | Discovered Policy | Discovered Role | Same-Seed Net Advantage | Same Class | Direct Boundary Count |",
                "| --- | --- | --- | --- | ---: | --- | ---: |",
            ]
        )
        for _, row in top_pairs.iterrows():
            rows.append(
                f"| `{row['classicFamily']}` | `{row['classicPolicyId']}` | `{row['discoveredPolicyId']}` | `{row['discoveredPrimaryRole']}` | "
                f"{float(row.get('sameSeedNetAdvantageScore', np.nan)):.3f} | {bool(row.get('sameUniversalityClass', False))} | "
                f"{int(row.get('directPhaseBoundaryCount', 0))} |"
            )
    rows.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Dominance and Pareto calls are computational proxy summaries over bounded S07-S13 evidence, not mathematical optimality proofs.",
            "- S13 uses literal same scheduler/tie seeds for its own CPU-reference panel, while upstream S09/S12 evidence is retained as context.",
            "- Discovered policies with no measured competence evidence are excluded from pairwise claims and written to missing-comparison records.",
        ]
    )
    return "\n".join(rows) + "\n"


def render_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    policy_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    run_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    selected = policy_df[policy_df["selectedForS13"]]
    classics = selected[selected["comparisonRole"].eq("classic")]
    discovered = selected[selected["comparisonRole"].eq("discovered")]
    family_counts = classics["classicFamily"].value_counts().to_dict()
    role_counts = discovered["primaryRole"].value_counts().to_dict()
    return "\n".join(
        [
            "# S13 Classics Vs Discovered Summary",
            "",
            "- Research step ID: S13",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_classics_vs_discovered.parquet'}`, `{STEP_DIR / 'classics_vs_discovered_report.md'}`, and `{FIGURES_DIR}/`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            "- Lay summary: S13 asks whether discovered local DSL policies dominate, complement, or sit near the classic Bubble/Insertion/Selection regions in the bounded morphospace evidence.",
            "",
            "## Key Counts",
            "",
            f"- Classic policies: {len(classics)} families `{canonical_json(family_counts)}`",
            f"- Discovered policies: {len(discovered)} roles `{canonical_json(role_counts)}`",
            f"- Same-seed CPU run rows: {len(run_df)}",
            f"- Pairwise classic/discovered comparisons: {len(pairwise_df)}",
            f"- Family summary rows: {len(family_summary_df)}",
        ]
    ) + "\n"


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[Path], caveats: list[str], recommended_next_action: str) -> str:
    rows = [
        "# S13 Validation Report",
        "",
        "- Research step ID: S13",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {int(validation_df['success'].sum())}/{len(validation_df)} checks passed",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "- Lay summary: S13 validation checks upstream anchors, classic-family coverage, discovered-policy coverage, same-seed runs and deltas, Pareto/front outputs, embeddings, phase boundaries, class placements, figures, tests, and absence of S14 artifacts.",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for _, row in validation_df.iterrows():
        rows.append(f"| `{row['checkId']}` | {'pass' if row['success'] else 'fail'} | {str(row['detail']).replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def copy_code_artifacts() -> list[Path]:
    paths: list[Path] = []
    for root in [STEP_DIR / "code", SHARED_CODE_DIR]:
        targets = [
            (REPO_ROOT / "scripts" / "e03_s13_classics_vs_discovered.py", root / "scripts" / "e03_s13_classics_vs_discovered.py"),
            (REPO_ROOT / "morphospace" / "classic_comparison.py", root / "morphospace" / "classic_comparison.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "morphospace" / "policies.py", root / "morphospace" / "policies.py"),
            (REPO_ROOT / "morphospace" / "rule_dsl.py", root / "morphospace" / "rule_dsl.py"),
            (REPO_ROOT / "tests" / "test_e03_classic_comparison.py", root / "tests" / "test_e03_classic_comparison.py"),
        ]
        for source, dest in targets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            paths.append(dest)
    return paths


def main() -> None:
    started_at = utc_now()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_written: list[Path] = []

    inputs = load_inputs()
    config = {
        "researchStepId": STEP_ID,
        "version": CLASSIC_COMPARISON_VERSION,
        "seedCount": SEED_COUNT,
        "classicFamilies": list(CLASSIC_FAMILIES),
        "tasks": [task.to_dict() for task in S13_TASKS],
        "inputs": {
            "corpus": str(CORPUS_PATH),
            "s08Elites": str(S08_ELITES_PATH),
            "s09Boundaries": str(S09_BOUNDARIES_PATH),
            "s10Features": str(S10_FEATURES_PATH),
            "s10Normalized": str(S10_NORMALIZED_PATH),
            "s10Embeddings": str(S10_EMBEDDINGS_PATH),
            "s10Neighbors": str(S10_NEIGHBORS_PATH),
            "s11Assignments": str(S11_ASSIGNMENTS_PATH),
            "s12PolicySpecs": str(S12_POLICY_SPECS_PATH),
            "s12Ablations": str(S12_ABLATIONS_PATH),
        },
    }
    config_path = STEP_DIR / "classic_comparison_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    initial_policy_df = build_policy_comparison_universe(
        features_df=inputs["s10_features"],
        assignments_df=inputs["s11_assignments"],
        corpus_df=inputs["corpus"],
        s12_policy_specs_df=inputs["s12_policy_specs"],
    )
    selected_for_run = initial_policy_df[initial_policy_df["selectedForS13"]].copy()
    selected_for_run = selected_for_run.merge(
        inputs["corpus"][["policyId", "dslProgramJson"]],
        on="policyId",
        how="left",
        validate="one_to_one",
    )
    if selected_for_run["dslProgramJson"].isna().any():
        missing = selected_for_run.loc[selected_for_run["dslProgramJson"].isna(), "policyId"].tolist()
        raise RuntimeError(f"S13 selected policies missing DSL programs: {missing[:10]}")

    selected_file_index_df = write_selected_policy_files(selected_for_run)
    if not bool(selected_file_index_df["roundtripSuccess"].all()):
        raise RuntimeError("S13 selected policy DSL roundtrip failed")

    expected_panel_rows = int(len(selected_for_run) * len(S13_TASKS) * SEED_COUNT)
    cached_run_path = STEP_DIR / "same_seed_runs.parquet"
    cached_vector_path = STEP_DIR / "same_seed_competence_vectors.parquet"
    used_cached_same_seed_panel = False
    if cached_run_path.exists() and cached_vector_path.exists():
        cached_run_df = pd.read_parquet(cached_run_path)
        cached_vector_df = pd.read_parquet(cached_vector_path)
        cached_policy_ids = set(cached_run_df.get("policyId", pd.Series(dtype=str)).astype(str))
        selected_policy_ids = set(selected_for_run["policyId"].astype(str))
        cache_version_ok = (
            "classicComparisonVersion" in cached_run_df.columns
            and set(cached_run_df["classicComparisonVersion"].astype(str)) == {CLASSIC_COMPARISON_VERSION}
        )
        if (
            len(cached_run_df) == expected_panel_rows
            and len(cached_vector_df) == expected_panel_rows
            and cached_policy_ids == selected_policy_ids
            and cache_version_ok
        ):
            run_df, vector_df = cached_run_df, cached_vector_df
            used_cached_same_seed_panel = True
        else:
            run_df, vector_df = run_same_seed_panel(selected_for_run)
    else:
        run_df, vector_df = run_same_seed_panel(selected_for_run)
    s13_metric_df = aggregate_s13_vectors(vector_df)
    policy_df = build_policy_comparison_universe(
        features_df=inputs["s10_features"],
        assignments_df=inputs["s11_assignments"],
        corpus_df=inputs["corpus"],
        s12_policy_specs_df=inputs["s12_policy_specs"],
        s13_policy_metrics_df=s13_metric_df,
    )
    missing_df = missing_comparison_records(policy_df)
    pareto_df = pareto_front_table(policy_df)
    task_delta_df = same_seed_task_delta_table(vector_df, policy_df)
    same_seed_pair_df = summarize_same_seed_pairs(task_delta_df)
    pairwise_df = build_pairwise_comparison_table(
        policy_df=policy_df,
        same_seed_summary_df=same_seed_pair_df,
        pareto_df=pareto_df,
        normalized_matrix_df=inputs["s10_normalized"],
        embeddings_df=inputs["s10_embeddings"],
        neighbors_df=inputs["s10_neighbors"],
        boundaries_df=inputs["s09_boundaries"],
    )
    family_summary_df = family_summary_table(pairwise_df, policy_df, pareto_df)

    artifacts_written.extend(write_table(policy_df, STEP_DIR / "policy_comparison_summary.csv", STEP_DIR / "policy_comparison_summary.parquet"))
    artifacts_written.extend(
        write_table(policy_df, RESULTS_DIR / "e03_s13_policy_comparison_summary.csv", RESULTS_DIR / "e03_s13_policy_comparison_summary.parquet")
    )
    artifacts_written.extend(write_table(pairwise_df, STEP_DIR / "classics_vs_discovered.csv", STEP_DIR / "classics_vs_discovered.parquet"))
    artifacts_written.extend(
        write_table(pairwise_df, RESULTS_DIR / "e03_classics_vs_discovered.csv", RESULTS_DIR / "e03_classics_vs_discovered.parquet")
    )
    artifacts_written.extend(write_table(run_df, STEP_DIR / "same_seed_runs.csv", STEP_DIR / "same_seed_runs.parquet"))
    artifacts_written.extend(write_table(run_df, RESULTS_DIR / "e03_s13_same_seed_runs.csv", RESULTS_DIR / "e03_s13_same_seed_runs.parquet"))
    artifacts_written.extend(write_table(vector_df, STEP_DIR / "same_seed_competence_vectors.csv", STEP_DIR / "same_seed_competence_vectors.parquet"))
    artifacts_written.extend(
        write_table(vector_df, RESULTS_DIR / "e03_s13_same_seed_competence_vectors.csv", RESULTS_DIR / "e03_s13_same_seed_competence_vectors.parquet")
    )
    artifacts_written.extend(write_table(task_delta_df, STEP_DIR / "same_seed_task_deltas.csv", STEP_DIR / "same_seed_task_deltas.parquet"))
    artifacts_written.extend(
        write_table(task_delta_df, RESULTS_DIR / "e03_s13_same_seed_task_deltas.csv", RESULTS_DIR / "e03_s13_same_seed_task_deltas.parquet")
    )
    artifacts_written.extend(write_table(same_seed_pair_df, STEP_DIR / "same_seed_pair_summary.csv", STEP_DIR / "same_seed_pair_summary.parquet"))
    artifacts_written.extend(
        write_table(same_seed_pair_df, RESULTS_DIR / "e03_s13_same_seed_pair_summary.csv", RESULTS_DIR / "e03_s13_same_seed_pair_summary.parquet")
    )
    artifacts_written.extend(write_table(pareto_df, STEP_DIR / "pareto_fronts.csv", STEP_DIR / "pareto_fronts.parquet"))
    artifacts_written.extend(write_table(pareto_df, RESULTS_DIR / "e03_s13_pareto_fronts.csv", RESULTS_DIR / "e03_s13_pareto_fronts.parquet"))
    artifacts_written.extend(write_table(family_summary_df, STEP_DIR / "family_comparison_summary.csv", STEP_DIR / "family_comparison_summary.parquet"))
    artifacts_written.extend(
        write_table(family_summary_df, RESULTS_DIR / "e03_s13_family_comparison_summary.csv", RESULTS_DIR / "e03_s13_family_comparison_summary.parquet")
    )
    artifacts_written.extend(write_table(missing_df, STEP_DIR / "missing_comparison_records.csv", STEP_DIR / "missing_comparison_records.parquet"))
    artifacts_written.extend(
        write_table(missing_df, RESULTS_DIR / "e03_s13_missing_comparison_records.csv", RESULTS_DIR / "e03_s13_missing_comparison_records.parquet")
    )
    artifacts_written.extend(write_table(selected_file_index_df, STEP_DIR / "selected_policy_file_index.csv", STEP_DIR / "selected_policy_file_index.parquet"))

    figure_paths = plot_outputs(
        policy_df=policy_df,
        pairwise_df=pairwise_df,
        family_summary_df=family_summary_df,
        embeddings_df=inputs["s10_embeddings"],
    )
    artifacts_written.extend(figure_paths)

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_e03_classic_comparison",
            "tests.test_e03_feature_ablations",
            "tests.test_e03_universality_classes",
            "tests.test_e03_behavior_embeddings",
            "tests.test_e03_phase_boundaries",
            "tests.test_e03_quality_diversity",
            "tests.test_e03_competence",
        ],
        timeout=900,
    )
    repo_log = STEP_DIR / "repo_unit_test_log.txt"
    repo_log.write_text(repo_test_payload["output"], encoding="utf-8")
    artifacts_written.append(repo_log)

    caveats = [
        "S13 comparisons are bounded computational proxy results over small S07-S13 panels, not mathematical optimality or biological evidence.",
        "The S13 same-seed panel uses identical scheduler and tie-breaker seeds, but upstream S09/S12 context retains its original policy-indexed seed design.",
        "Discovered policies without measured competence evidence are excluded from pairwise claims and recorded as missing-comparison records.",
        "Pareto calls depend on the selected normalized score dimensions and should guide S14 candidate curation rather than replace independent holdout validation.",
    ]
    recommended_next_action = "Stop before S14 for Chief Scientist review; if approved, curate frontier candidates using S13 dominance/Pareto/class-placement evidence."

    comparison_report_path = STEP_DIR / "classics_vs_discovered_report.md"
    comparison_report_path.write_text(
        render_comparison_report(
            validation_result="pending until final validation table is written",
            artifacts_written=artifacts_written,
            policy_df=policy_df,
            pairwise_df=pairwise_df,
            family_summary_df=family_summary_df,
            pareto_df=pareto_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    artifacts_written.append(comparison_report_path)

    upstream_statuses = {
        step: load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json") for step in ["S08", "S09", "S10", "S11", "S12"]
    }
    validation_df = validate_classic_comparison_outputs(
        policy_df=policy_df,
        pairwise_df=pairwise_df,
        task_delta_df=task_delta_df,
        pareto_df=pareto_df,
        family_summary_df=family_summary_df,
        upstream_statuses=upstream_statuses,
        repo_test_payload={key: value for key, value in repo_test_payload.items() if key != "output"},
        report_exists=comparison_report_path.exists(),
        figure_paths=[str(path) for path in figure_paths if path.exists()],
        s14_dir_exists=(ARTIFACTS_DIR / "research_steps" / "S14").exists(),
    )
    expected_run_rows = int(policy_df["selectedForS13"].sum()) * len(S13_TASKS) * SEED_COUNT
    validation_df = pd.concat(
        [
            validation_df,
            pd.DataFrame(
                [
                    {
                        "checkId": "same_seed_run_rows_complete",
                        "success": len(run_df) == expected_run_rows,
                        "detail": f"rows={len(run_df)} expected={expected_run_rows}",
                    },
                    {
                        "checkId": "run_value_counts_conserved",
                        "success": bool(run_df["valueCountsConserved"].all()),
                        "detail": "all S13 CPU runs preserved input value multisets",
                    },
                    {
                        "checkId": "selected_policy_file_integrity",
                        "success": bool(
                            selected_file_index_df["dslPath"].map(lambda path: Path(str(path)).exists()).all()
                            and selected_file_index_df.apply(lambda row: sha256_file(Path(str(row["dslPath"]))) == row["dslSha256"], axis=1).all()
                        ),
                        "detail": f"{len(selected_file_index_df)} checksummed selected policy DSL files",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "classic_comparison_validation.csv", STEP_DIR / "classic_comparison_validation.parquet"))
    artifacts_written.extend(
        write_table(validation_df, RESULTS_DIR / "e03_s13_classic_comparison_validation.csv", RESULTS_DIR / "e03_s13_classic_comparison_validation.parquet")
    )

    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )

    code_paths = copy_code_artifacts()
    artifacts_written.extend(code_paths)
    summary_path = STEP_DIR / "summary.md"
    validation_report_path = STEP_DIR / "validation_report.md"
    status_path = STEP_DIR / "status.json"
    manifest_path = STEP_DIR / "artifact_manifest.json"
    final_artifact_paths = artifacts_written + [summary_path, validation_report_path, status_path, manifest_path]
    summary_path.write_text(
        render_summary(
            validation_result=validation_result,
            artifacts_written=final_artifact_paths,
            policy_df=policy_df,
            pairwise_df=pairwise_df,
            run_df=run_df,
            family_summary_df=family_summary_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    validation_report_path.write_text(
        render_validation_report(validation_df, final_artifact_paths, caveats, recommended_next_action),
        encoding="utf-8",
    )
    artifacts_written.extend([summary_path, validation_report_path])
    comparison_report_path.write_text(
        render_comparison_report(
            validation_result=validation_result,
            artifacts_written=final_artifact_paths,
            policy_df=policy_df,
            pairwise_df=pairwise_df,
            family_summary_df=family_summary_df,
            pareto_df=pareto_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )

    git_metadata = get_git_metadata()
    selected = policy_df[policy_df["selectedForS13"]]
    classics = selected[selected["comparisonRole"].eq("classic")]
    discovered = selected[selected["comparisonRole"].eq("discovered")]
    effect_summary = {
        "classicPolicyCount": int(len(classics)),
        "classicFamilyCounts": {str(key): int(value) for key, value in classics["classicFamily"].value_counts().to_dict().items()},
        "discoveredPolicyCount": int(len(discovered)),
        "discoveredRoleCounts": {str(key): int(value) for key, value in discovered["primaryRole"].value_counts().to_dict().items()},
        "selectedPolicyCount": int(len(selected)),
        "sameSeedRunRows": int(len(run_df)),
        "sameSeedVectorRows": int(len(vector_df)),
        "sameSeedTaskDeltaRows": int(len(task_delta_df)),
        "pairwiseComparisonRows": int(len(pairwise_df)),
        "paretoFrontRows": int(len(pareto_df)),
        "paretoOptimalPolicyCount": int(pareto_df["isParetoOptimal"].sum()) if "isParetoOptimal" in pareto_df else 0,
        "familySummaryRows": int(len(family_summary_df)),
        "missingComparisonRecords": int(len(missing_df)),
        "discoveredDominatingBroadPairCount": int(pairwise_df["discoveredDominatesClassicBroad"].sum()),
        "discoveredDominatingSameSeedPairCount": int(pairwise_df["discoveredDominatesClassicSameSeed"].fillna(False).sum()),
    }
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": TITLE,
        "success": validation_passed,
        "status": STATUS if validation_passed else "completed_with_validation_failures",
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
        "artifactsWritten": [str(path) for path in artifacts_written] + [str(status_path), str(manifest_path)],
        "validationResult": validation_result,
        "validationChecksPassed": int(validation_df["success"].sum()),
        "validationChecksTotal": int(len(validation_df)),
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "laySummary": (
            "S13 compared Bubble, Insertion, and Selection-family policies against evidence-supported discovered DSL policies using "
            "same-seed CPU probes plus embedding, Pareto, phase-boundary, and class-placement context."
        ),
        "classicComparisonSummary": effect_summary,
        "usedCachedSameSeedPanel": bool(used_cached_same_seed_panel),
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {"classicComparisonVersion": CLASSIC_COMPARISON_VERSION},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
        },
        "git": git_metadata,
        "startedAt": started_at,
        "completedAt": utc_now(),
    }
    write_json(status_path, status_payload)
    artifacts_written.append(status_path)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_passed,
        "status": status_payload["status"],
        "artifactsWritten": [str(path) for path in artifacts_written] + [str(manifest_path)],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "artifactCount": len(artifacts_written) + 1,
        "checksummedArtifactCount": len(artifact_rows(artifacts_written)),
        "artifacts": artifact_rows(artifacts_written),
        "selectedPolicyFileCount": int(len(selected_file_index_df)),
        "selectedPolicyFileIndexPath": str(STEP_DIR / "selected_policy_file_index.parquet"),
        "manifestSelfReference": {"path": str(manifest_path), "sha256": "omitted_self_referential_manifest"},
        "git": git_metadata,
        "versions": status_payload["versions"],
    }
    write_json(manifest_path, manifest_payload)
    artifacts_written.append(manifest_path)

    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": validation_passed,
                "validationResult": validation_result,
                "classicPolicies": effect_summary["classicPolicyCount"],
                "discoveredPolicies": effect_summary["discoveredPolicyCount"],
                "sameSeedRunRows": effect_summary["sameSeedRunRows"],
                "pairwiseComparisonRows": effect_summary["pairwiseComparisonRows"],
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
