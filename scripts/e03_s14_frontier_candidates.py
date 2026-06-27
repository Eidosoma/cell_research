#!/usr/bin/env python3
"""Execute E03 S14: pick and validate frontier candidate policies.

S14 uses the completed S08-S13 morphospace evidence to curate 10-20 novel DSL
policies for downstream atlas, memory/repair, and chimera experiments.  It then
runs a bounded CPU-reference holdout panel with independent seeds and larger
arrays, writes checksummed DSL files, and stops before S15.
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
    FRONTIER_CANDIDATE_VERSION,
    FrontierValidationTask,
    DSLPolicy,
    NullPolicy,
    PolicyEventSimulator,
    build_frontier_candidate_pool,
    compute_competence_vector,
    parse_rule_program,
    select_frontier_candidates,
    summarize_frontier_validation,
    validate_frontier_outputs,
)
from morphospace.competence import canonical_json, delayed_gratification_from_sortedness, sortedness_sign_change_count  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S14"
STEP_NUMBER = 14
TITLE = "Pick frontier candidates"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
SEED_COUNT = 4
TARGET_CANDIDATE_COUNT = 16

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = STEP_DIR / "figures"
FRONTIER_POLICY_DIR = STEP_DIR / "frontier_policy_dsl"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_frontier_candidates"

CORPUS_PATH = ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet"
S08_ELITES_PATH = RESULTS_DIR / "e03_qd_elites.parquet"
S09_BOUNDARIES_PATH = RESULTS_DIR / "e03_phase_boundaries.parquet"
S11_ASSIGNMENTS_PATH = RESULTS_DIR / "e03_s11_policy_class_assignments.parquet"
S12_ABLATIONS_PATH = RESULTS_DIR / "e03_feature_ablations.parquet"
S13_POLICY_SUMMARY_PATH = RESULTS_DIR / "e03_s13_policy_comparison_summary.parquet"
S13_PAIRWISE_PATH = RESULTS_DIR / "e03_classics_vs_discovered.parquet"
S13_PARETO_PATH = RESULTS_DIR / "e03_s13_pareto_fronts.parquet"
S13_RUNS_PATH = RESULTS_DIR / "e03_s13_same_seed_runs.parquet"

S14_TASKS = (
    FrontierValidationTask(
        task_id="s14_holdout_unique_n8",
        task_family="sorting",
        task_panel="s14_frontier_holdout_larger_array",
        input_profile="unique_n8_interleaved_holdout",
        initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
        scheduler_seed_base=81000,
        tie_seed_base=91000,
        max_activations=1800,
        max_swaps=1800,
        max_comparisons=7200,
    ),
    FrontierValidationTask(
        task_id="s14_holdout_duplicate_n8",
        task_family="sorting_duplicate_values",
        task_panel="s14_frontier_holdout_larger_array",
        input_profile="duplicate_n8_holdout",
        initial_values=(4, 2, 4, 1, 3, 1, 2, 3),
        scheduler_seed_base=82000,
        tie_seed_base=92000,
        max_activations=1800,
        max_swaps=1800,
        max_comparisons=7200,
    ),
    FrontierValidationTask(
        task_id="s14_holdout_stuck_frozen_n8",
        task_family="frozen",
        task_panel="s14_frontier_holdout_larger_array",
        input_profile="unique_n8_stuck_frozen_holdout",
        initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
        scheduler_seed_base=83000,
        tie_seed_base=93000,
        frozen_variant="stuck",
        frozen_positions=(3,),
        max_activations=2200,
        max_swaps=2200,
        max_comparisons=8800,
    ),
    FrontierValidationTask(
        task_id="s14_holdout_transfer_n10",
        task_family="transfer",
        task_panel="s14_frontier_holdout_larger_array",
        input_profile="unique_n10_transfer_holdout",
        initial_values=(10, 1, 9, 2, 8, 3, 7, 4, 6, 5),
        scheduler_seed_base=84000,
        tie_seed_base=94000,
        max_activations=3000,
        max_swaps=3000,
        max_comparisons=12000,
    ),
    FrontierValidationTask(
        task_id="s14_holdout_chimera_null_n10",
        task_family="chimera",
        task_panel="s14_frontier_holdout_larger_array",
        input_profile="alternating_dsl_null_chimera_n10_holdout",
        initial_values=(10, 1, 9, 2, 8, 3, 7, 4, 6, 5),
        scheduler_seed_base=85000,
        tie_seed_base=95000,
        chimera_with_null=True,
        candidate_positions=(0, 2, 4, 6, 8),
        max_activations=3000,
        max_swaps=3000,
        max_comparisons=12000,
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
        "s11_assignments": S11_ASSIGNMENTS_PATH,
        "s12_ablations": S12_ABLATIONS_PATH,
        "s13_policy_summary": S13_POLICY_SUMMARY_PATH,
        "s13_pairwise": S13_PAIRWISE_PATH,
        "s13_pareto": S13_PARETO_PATH,
        "s13_runs": S13_RUNS_PATH,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"S14 requires completed upstream inputs; missing: {missing}")
    for step in ["S08", "S09", "S10", "S11", "S12", "S13"]:
        status = load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        if not status.get("success"):
            raise RuntimeError(f"S14 requires successful {step}; got {status}")
    return {key: pd.read_parquet(path) for key, path in required_paths.items()}


def value_counts_conserved(initial_values: tuple[int, ...], final_values: list[int]) -> bool:
    return Counter(map(int, initial_values)) == Counter(map(int, final_values))


def trace_sortedness(result) -> list[float]:
    return [float(row["sortedness_percent"]) for row in result.trace_rows]


def trace_hashes(result) -> list[str]:
    return [str(row["state_hash"]) for row in result.trace_rows]


def make_policies_for_task(program, task: FrontierValidationTask):
    if not task.chimera_with_null:
        return DSLPolicy(program)
    candidate_positions = set(map(int, task.candidate_positions))
    return [DSLPolicy(program) if position in candidate_positions else NullPolicy() for position in range(len(task.initial_values))]


def run_one_frontier_task_seed(row: pd.Series, task: FrontierValidationTask, seed_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_id = str(row["policyId"])
    program = parse_rule_program(row["dslProgramJson"])
    scheduler_seed = int(task.scheduler_seed_base + int(seed_index))
    tie_seed = int(task.tie_seed_base + int(seed_index))
    condition_id = f"S14::{task.task_id}::{policy_id}::seed{scheduler_seed}"
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
        implementation="e03_s14_cpu_frontier_holdout",
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
        "frontierCandidateVersion": FRONTIER_CANDIDATE_VERSION,
        "researchStepId": STEP_ID,
        "policyId": policy_id,
        "frontierRank": int(row.get("frontierRank", -1)),
        "frontierObjectiveTagsJson": str(row.get("frontierObjectiveTagsJson", "[]")),
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
        source_metric_source="s14_cpu_frontier_holdout",
        source_artifact_path=str(STEP_DIR / "frontier_validation_runs.parquet"),
        transfer_score=final_sortedness_score if task.task_family == "transfer" else None,
        compatibility_score=final_aggregation if task.task_family == "chimera" else None,
    )
    vector.update(
        {
            "frontierCandidateVersion": FRONTIER_CANDIDATE_VERSION,
            "frontierRank": int(row.get("frontierRank", -1)),
            "frontierObjectiveTagsJson": str(row.get("frontierObjectiveTagsJson", "[]")),
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


def run_holdout_panel(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    sorted_candidates = candidate_df.sort_values(["frontierRank", "policyId"], kind="mergesort")
    for _, row in sorted_candidates.iterrows():
        for task in S14_TASKS:
            for seed_index in range(SEED_COUNT):
                run_record, vector = run_one_frontier_task_seed(row, task, seed_index)
                run_rows.append(run_record)
                vector_rows.append(vector)
    return pd.DataFrame(run_rows), pd.DataFrame(vector_rows)


def write_frontier_policy_files(candidate_df: pd.DataFrame) -> pd.DataFrame:
    FRONTIER_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in candidate_df.sort_values(["frontierRank", "policyId"], kind="mergesort").iterrows():
        program = parse_rule_program(row["dslProgramJson"])
        policy_id = str(row["policyId"])
        path = FRONTIER_POLICY_DIR / f"{int(row['frontierRank']):02d}_{policy_id}.json"
        path.write_text(json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "frontierCandidateVersion": FRONTIER_CANDIDATE_VERSION,
                "policyId": policy_id,
                "frontierRank": int(row["frontierRank"]),
                "className": str(row.get("className", "")),
                "frontierObjectiveTagsJson": str(row.get("frontierObjectiveTagsJson", "[]")),
                "dslPath": str(path),
                "dslSha256": sha256_file(path),
                "roundtripSuccess": canonical_json(parse_rule_program(program.to_json()).to_dict()) == canonical_json(program.to_dict()),
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(pool_df: pd.DataFrame, candidate_df: pd.DataFrame, validation_summary_df: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    selected_ids = set(candidate_df["policyId"].astype(str)) if "policyId" in candidate_df else set()

    fig, ax = plt.subplots(figsize=(8, 6))
    pool_plot = pool_df.copy()
    ax.scatter(
        pool_plot["s13CompositeScore"],
        pool_plot["frontierCompositeScore"],
        s=14,
        alpha=0.35,
        color="#6b7280",
        label="candidate pool",
    )
    selected_plot = pool_plot[pool_plot["policyId"].astype(str).isin(selected_ids)]
    if not selected_plot.empty:
        ax.scatter(
            selected_plot["s13CompositeScore"],
            selected_plot["frontierCompositeScore"],
            s=42,
            color="#2f5f8f",
            edgecolor="black",
            linewidth=0.5,
            label="selected frontier",
        )
    ax.set_xlabel("S13 broad composite score")
    ax.set_ylabel("S14 frontier composite score")
    ax.set_title("S14 frontier selection surface")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES_DIR / "s14_frontier_selection_surface.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    if not validation_summary_df.empty:
        summary = validation_summary_df.sort_values("frontierRank", kind="mergesort")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        labels = summary["frontierRank"].astype(int).astype(str) + ": " + summary["policyId"].astype(str).str[-6:]
        ax.bar(labels, summary["holdoutCompositeScore"], color="#687d45")
        ax.set_xlabel("frontier rank and policy suffix")
        ax.set_ylabel("holdout composite score")
        ax.set_title("S14 independent-seed larger-array validation")
        ax.tick_params(axis="x", rotation=55)
        fig.tight_layout()
        path = FIGURES_DIR / "s14_holdout_composite_by_candidate.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        task_cols = [column for column in summary.columns if column.startswith("holdout_completionRateTaskFamily_")]
        if task_cols:
            heat = summary[task_cols].to_numpy(dtype=float)
            im = ax.imshow(heat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_yticks(np.arange(len(summary)), summary["frontierRank"].astype(int).astype(str))
            ax.set_xticks(np.arange(len(task_cols)), [column.removeprefix("holdout_completionRateTaskFamily_") for column in task_cols], rotation=35, ha="right")
            ax.set_ylabel("frontier rank")
            ax.set_title("S14 holdout completion by task family")
            fig.colorbar(im, ax=ax, label="completion rate")
            fig.tight_layout()
            path = FIGURES_DIR / "s14_holdout_completion_heatmap.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            paths.append(path)
        else:
            plt.close(fig)
    return paths


def render_frontier_report(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    candidate_df: pd.DataFrame,
    validation_summary_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    top = validation_summary_df.sort_values("holdoutCompositeScore", ascending=False, kind="mergesort").head(8)
    rows = [
        "# S14 Frontier Candidate Report",
        "",
        "- Research step ID: S14",
        f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
        f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_frontier_candidates.parquet'}`, `{STEP_DIR / 'frontier_policy_dsl'}/`, and `{RESULTS_DIR / 'e03_s14_frontier_validation_summary.parquet'}`",
        f"- Validation result: {validation_result}",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "- Lay summary: S14 chooses a small set of novel local-rule policies from the prior evidence and checks them on unseen seeds with larger arrays before they become atlas and downstream-experiment candidates.",
        "",
        "## Curation Design",
        "",
        "- Inputs: S08 QD elites, S09 phase-boundary records, S11 universality classes, S12 ablation context, and S13 classic/discovered comparisons.",
        "- Exclusions: classic Bubble/Insertion/Selection-family policies and null policies are excluded from the frontier set.",
        "- Diversity controls: exact duplicate DSL structure hashes are excluded, and objective tags/classes are intentionally diversified.",
        "- Holdout panel: five CPU-reference task families use independent scheduler/tie seeds beginning at 81000/91000 and arrays of length 8 or 10, larger than S13's length-6 maximum.",
        "",
        "## Selected Candidates",
        "",
        "| Rank | Policy | Class | Tags | Frontier Score | Selection Stage |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for _, row in candidate_df.sort_values("frontierRank", kind="mergesort").iterrows():
        tags = ", ".join(json.loads(row["frontierObjectiveTagsJson"]))
        rows.append(
            f"| {int(row['frontierRank'])} | `{row['policyId']}` | `{row.get('className', '')}` | {tags} | "
            f"{float(row.get('frontierCompositeScore', np.nan)):.3f} | `{row.get('frontierSelectionStage', '')}` |"
        )
    rows.extend(["", "## Holdout Highlights", ""])
    if top.empty:
        rows.append("- Holdout validation summary was empty.")
    else:
        rows.extend(
            [
                "| Rank | Policy | Holdout Composite | Completion Mean | Transfer Completion | Chimera Completion |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in top.iterrows():
            rows.append(
                f"| {int(row.get('frontierRank', -1))} | `{row['policyId']}` | "
                f"{float(row.get('holdoutCompositeScore', np.nan)):.3f} | "
                f"{float(row.get('holdout_completionSuccessMean', np.nan)):.3f} | "
                f"{float(row.get('holdout_completionRateTaskFamily_transfer', np.nan)):.3f} | "
                f"{float(row.get('holdout_completionRateTaskFamily_chimera', np.nan)):.3f} |"
            )
    rows.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Candidate status means bounded computational promise, not proof of optimality.",
            "- Holdout runs are stronger than the S13 same-seed comparison for these candidates, but they are still small CPU-reference panels.",
            "- S15 should present these candidates with their caveats, failure modes, and links to exact DSL files rather than as definitive winners.",
        ]
    )
    return "\n".join(rows) + "\n"


def render_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    candidate_df: pd.DataFrame,
    validation_run_df: pd.DataFrame,
    validation_summary_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    objective_tags = sorted({tag for value in candidate_df["frontierObjectiveTagsJson"] for tag in json.loads(value)})
    class_counts = candidate_df["className"].astype(str).value_counts().to_dict()
    return "\n".join(
        [
            "# S14 Frontier Candidates Summary",
            "",
            "- Research step ID: S14",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_frontier_candidates.parquet'}`, `{STEP_DIR / 'frontier_candidates_report.md'}`, and `{FIGURES_DIR}/`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            "- Lay summary: S14 narrows the large morphospace to a small, novel frontier set and stress-checks it on unseen seeds and larger arrays.",
            "",
            "## Key Counts",
            "",
            f"- Selected frontier candidates: {len(candidate_df)}",
            f"- Objective tags represented: `{canonical_json(objective_tags)}`",
            f"- Universality classes represented: `{canonical_json(class_counts)}`",
            f"- Holdout CPU run rows: {len(validation_run_df)}",
            f"- Holdout validation summary rows: {len(validation_summary_df)}",
        ]
    ) + "\n"


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[Path], caveats: list[str], recommended_next_action: str) -> str:
    rows = [
        "# S14 Validation Report",
        "",
        "- Research step ID: S14",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {int(validation_df['success'].sum())}/{len(validation_df)} checks passed",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "- Lay summary: S14 validation checks upstream anchors, novel candidate count, DSL integrity, duplicate structure hashes, objective/class diversity, independent larger-array holdout rows, reports, figures, tests, and absence of S15 artifacts.",
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
            (REPO_ROOT / "scripts" / "e03_s14_frontier_candidates.py", root / "scripts" / "e03_s14_frontier_candidates.py"),
            (REPO_ROOT / "morphospace" / "frontier_candidates.py", root / "morphospace" / "frontier_candidates.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "morphospace" / "policies.py", root / "morphospace" / "policies.py"),
            (REPO_ROOT / "morphospace" / "rule_dsl.py", root / "morphospace" / "rule_dsl.py"),
            (REPO_ROOT / "tests" / "test_e03_frontier_candidates.py", root / "tests" / "test_e03_frontier_candidates.py"),
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
        "version": FRONTIER_CANDIDATE_VERSION,
        "seedCount": SEED_COUNT,
        "targetCandidateCount": TARGET_CANDIDATE_COUNT,
        "tasks": [task.to_dict() for task in S14_TASKS],
        "inputs": {
            "corpus": str(CORPUS_PATH),
            "s08Elites": str(S08_ELITES_PATH),
            "s09Boundaries": str(S09_BOUNDARIES_PATH),
            "s11Assignments": str(S11_ASSIGNMENTS_PATH),
            "s12Ablations": str(S12_ABLATIONS_PATH),
            "s13PolicySummary": str(S13_POLICY_SUMMARY_PATH),
            "s13Pairwise": str(S13_PAIRWISE_PATH),
            "s13Pareto": str(S13_PARETO_PATH),
            "s13Runs": str(S13_RUNS_PATH),
        },
    }
    config_path = STEP_DIR / "frontier_candidate_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    pool_df = build_frontier_candidate_pool(
        policy_df=inputs["s13_policy_summary"],
        pairwise_df=inputs["s13_pairwise"],
        pareto_df=inputs["s13_pareto"],
        corpus_df=inputs["corpus"],
        qd_elites_df=inputs["s08_elites"],
        phase_boundaries_df=inputs["s09_boundaries"],
        class_assignments_df=inputs["s11_assignments"],
        feature_ablations_df=inputs["s12_ablations"],
    )
    candidate_df = select_frontier_candidates(pool_df, target_count=TARGET_CANDIDATE_COUNT)
    if not (10 <= len(candidate_df) <= 20):
        raise RuntimeError(f"S14 selected {len(candidate_df)} candidates; expected 10-20")
    if candidate_df["dslProgramJson"].isna().any():
        missing = candidate_df.loc[candidate_df["dslProgramJson"].isna(), "policyId"].tolist()
        raise RuntimeError(f"S14 selected policies missing DSL programs: {missing[:10]}")

    dsl_index_df = write_frontier_policy_files(candidate_df)
    if not bool(dsl_index_df["roundtripSuccess"].all()):
        raise RuntimeError("S14 frontier policy DSL roundtrip failed")

    expected_panel_rows = int(len(candidate_df) * len(S14_TASKS) * SEED_COUNT)
    cached_run_path = STEP_DIR / "frontier_validation_runs.parquet"
    cached_vector_path = STEP_DIR / "frontier_validation_competence_vectors.parquet"
    used_cached_holdout_panel = False
    if cached_run_path.exists() and cached_vector_path.exists():
        cached_run_df = pd.read_parquet(cached_run_path)
        cached_vector_df = pd.read_parquet(cached_vector_path)
        cached_policy_ids = set(cached_run_df.get("policyId", pd.Series(dtype=str)).astype(str))
        selected_policy_ids = set(candidate_df["policyId"].astype(str))
        cache_version_ok = (
            "frontierCandidateVersion" in cached_run_df.columns
            and set(cached_run_df["frontierCandidateVersion"].astype(str)) == {FRONTIER_CANDIDATE_VERSION}
        )
        if (
            len(cached_run_df) == expected_panel_rows
            and len(cached_vector_df) == expected_panel_rows
            and cached_policy_ids == selected_policy_ids
            and cache_version_ok
        ):
            validation_run_df, validation_vector_df = cached_run_df, cached_vector_df
            used_cached_holdout_panel = True
        else:
            validation_run_df, validation_vector_df = run_holdout_panel(candidate_df)
    else:
        validation_run_df, validation_vector_df = run_holdout_panel(candidate_df)
    validation_summary_df = summarize_frontier_validation(validation_vector_df, validation_run_df, candidate_df)

    rationale_cols = [
        "frontierRank",
        "policyId",
        "frontierSelectionStage",
        "frontierObjectiveTagsJson",
        "selectionRationaleJson",
        "frontierEvidenceJson",
        "frontierCompositeScore",
        "s13CompositeScore",
        "paretoFrontRank",
        "isParetoOptimal",
        "s13DominatesSameSeedCount",
        "s13ClassicDominatesSameSeedCount",
        "className",
        "classLabel",
        "lineageId",
        "generationMethod",
        "structureHash",
        "duplicateOrNearDuplicateJustification",
    ]
    rationale_df = candidate_df[[column for column in rationale_cols if column in candidate_df.columns]].copy()

    artifacts_written.extend(write_table(pool_df, STEP_DIR / "frontier_candidate_pool.csv", STEP_DIR / "frontier_candidate_pool.parquet"))
    artifacts_written.extend(
        write_table(pool_df, RESULTS_DIR / "e03_s14_frontier_candidate_pool.csv", RESULTS_DIR / "e03_s14_frontier_candidate_pool.parquet")
    )
    artifacts_written.extend(write_table(candidate_df, STEP_DIR / "frontier_candidates.csv", STEP_DIR / "frontier_candidates.parquet"))
    artifacts_written.extend(write_table(candidate_df, RESULTS_DIR / "e03_frontier_candidates.csv", RESULTS_DIR / "e03_frontier_candidates.parquet"))
    artifacts_written.extend(write_table(rationale_df, STEP_DIR / "frontier_selection_rationale.csv", STEP_DIR / "frontier_selection_rationale.parquet"))
    artifacts_written.extend(
        write_table(rationale_df, RESULTS_DIR / "e03_s14_frontier_selection_rationale.csv", RESULTS_DIR / "e03_s14_frontier_selection_rationale.parquet")
    )
    artifacts_written.extend(write_table(validation_run_df, STEP_DIR / "frontier_validation_runs.csv", STEP_DIR / "frontier_validation_runs.parquet"))
    artifacts_written.extend(
        write_table(validation_run_df, RESULTS_DIR / "e03_s14_frontier_validation_runs.csv", RESULTS_DIR / "e03_s14_frontier_validation_runs.parquet")
    )
    artifacts_written.extend(
        write_table(validation_vector_df, STEP_DIR / "frontier_validation_competence_vectors.csv", STEP_DIR / "frontier_validation_competence_vectors.parquet")
    )
    artifacts_written.extend(
        write_table(
            validation_vector_df,
            RESULTS_DIR / "e03_s14_frontier_validation_competence_vectors.csv",
            RESULTS_DIR / "e03_s14_frontier_validation_competence_vectors.parquet",
        )
    )
    artifacts_written.extend(write_table(validation_summary_df, STEP_DIR / "frontier_validation_summary.csv", STEP_DIR / "frontier_validation_summary.parquet"))
    artifacts_written.extend(
        write_table(
            validation_summary_df,
            RESULTS_DIR / "e03_s14_frontier_validation_summary.csv",
            RESULTS_DIR / "e03_s14_frontier_validation_summary.parquet",
        )
    )
    artifacts_written.extend(write_table(dsl_index_df, STEP_DIR / "frontier_policy_file_index.csv", STEP_DIR / "frontier_policy_file_index.parquet"))

    figure_paths = plot_outputs(pool_df, candidate_df, validation_summary_df)
    artifacts_written.extend(figure_paths)

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_e03_frontier_candidates",
            "tests.test_e03_classic_comparison",
            "tests.test_e03_competence",
            "tests.test_e03_rule_dsl",
        ],
        timeout=900,
    )
    repo_log = STEP_DIR / "repo_unit_test_log.txt"
    repo_log.write_text(repo_test_payload["output"], encoding="utf-8")
    artifacts_written.append(repo_log)

    caveats = [
        "S14 frontier scores are bounded computational curation weights over S08-S13 proxy evidence, not mathematical optimality.",
        "The holdout panel uses independent seeds and larger arrays but remains a small CPU-reference screen, not a high-power scaling study.",
        "Chimeric compatibility is represented by alternating DSL/null arrays, so it is a proxy for later E06 chimeric-governance experiments.",
        "Exact duplicate DSL structures are excluded; near-duplicate behavior can still occur and should be displayed with neighbors in S15.",
    ]
    recommended_next_action = "Stop before S15 for Chief Scientist review; if approved, build the atlas using these frontier candidates with their validation caveats and DSL links."

    report_path = STEP_DIR / "frontier_candidates_report.md"
    report_path.write_text(
        render_frontier_report(
            validation_result="pending until final validation table is written",
            artifacts_written=artifacts_written,
            candidate_df=candidate_df,
            validation_summary_df=validation_summary_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    artifacts_written.append(report_path)

    upstream_statuses = {
        step: load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        for step in ["S08", "S09", "S10", "S11", "S12", "S13"]
    }
    validation_df = validate_frontier_outputs(
        candidate_df=candidate_df,
        pool_df=pool_df,
        validation_run_df=validation_run_df,
        validation_vector_df=validation_vector_df,
        validation_summary_df=validation_summary_df,
        dsl_index_df=dsl_index_df,
        upstream_statuses=upstream_statuses,
        repo_test_payload={key: value for key, value in repo_test_payload.items() if key != "output"},
        report_exists=report_path.exists(),
        figure_paths=[str(path) for path in figure_paths if path.exists()],
        s15_dir_exists=(ARTIFACTS_DIR / "research_steps" / "S15").exists(),
        expected_run_rows=expected_panel_rows,
    )
    validation_df = pd.concat(
        [
            validation_df,
            pd.DataFrame(
                [
                    {
                        "checkId": "frontier_policy_file_integrity",
                        "success": bool(
                            dsl_index_df["dslPath"].map(lambda path: Path(str(path)).exists()).all()
                            and dsl_index_df.apply(lambda row: sha256_file(Path(str(row["dslPath"]))) == row["dslSha256"], axis=1).all()
                        ),
                        "detail": f"{len(dsl_index_df)} checksummed frontier policy DSL files",
                    },
                    {
                        "checkId": "holdout_summary_has_finite_scores",
                        "success": bool(pd.to_numeric(validation_summary_df["holdoutCompositeScore"], errors="coerce").notna().any()),
                        "detail": "at least one finite holdout composite score is present",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "frontier_candidate_validation.csv", STEP_DIR / "frontier_candidate_validation.parquet"))
    artifacts_written.extend(
        write_table(validation_df, RESULTS_DIR / "e03_s14_frontier_candidate_validation.csv", RESULTS_DIR / "e03_s14_frontier_candidate_validation.parquet")
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
            candidate_df=candidate_df,
            validation_run_df=validation_run_df,
            validation_summary_df=validation_summary_df,
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
    report_path.write_text(
        render_frontier_report(
            validation_result=validation_result,
            artifacts_written=final_artifact_paths,
            candidate_df=candidate_df,
            validation_summary_df=validation_summary_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )

    git_metadata = get_git_metadata()
    effect_summary = {
        "candidateCount": int(len(candidate_df)),
        "candidatePolicyIds": candidate_df.sort_values("frontierRank")["policyId"].astype(str).tolist(),
        "objectiveTags": sorted({tag for value in candidate_df["frontierObjectiveTagsJson"] for tag in json.loads(value)}),
        "classCounts": {str(key): int(value) for key, value in candidate_df["className"].astype(str).value_counts().to_dict().items()},
        "holdoutRunRows": int(len(validation_run_df)),
        "holdoutVectorRows": int(len(validation_vector_df)),
        "holdoutSummaryRows": int(len(validation_summary_df)),
        "holdoutTaskFamilies": sorted(set(validation_run_df["taskFamily"].astype(str))),
        "holdoutCompletionRateMean": float(pd.to_numeric(validation_summary_df["holdout_completionSuccessMean"], errors="coerce").mean()),
        "bestHoldoutPolicyId": str(
            validation_summary_df.sort_values("holdoutCompositeScore", ascending=False, kind="mergesort").iloc[0]["policyId"]
        )
        if not validation_summary_df.empty
        else "",
        "usedCachedHoldoutPanel": bool(used_cached_holdout_panel),
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
            "S14 curated a small set of novel frontier DSL policies from S08-S13 evidence and validated them on "
            "independent-seed larger-array CPU-reference panels."
        ),
        "frontierCandidateSummary": effect_summary,
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {"frontierCandidateVersion": FRONTIER_CANDIDATE_VERSION},
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
        "frontierPolicyFileCount": int(len(dsl_index_df)),
        "frontierPolicyFileIndexPath": str(STEP_DIR / "frontier_policy_file_index.parquet"),
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
                "candidateCount": effect_summary["candidateCount"],
                "holdoutRunRows": effect_summary["holdoutRunRows"],
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
