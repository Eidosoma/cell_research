#!/usr/bin/env python3
"""Execute E03 S09: find bounded phase-boundary candidates.

S09 consumes the completed S08 elite set and S05 parameterized DSL corpus,
then runs repeated-seed CPU reference probes on small tasks that expose sorting
success, delayed gratification, aggregation, oscillation, and frozen-cell
robustness transitions.  It stops before S10.
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
from dataclasses import dataclass
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
    DSLPolicy,
    PHASE_BOUNDARY_VERSION,
    PhaseBoundaryTask,
    PolicyEventSimulator,
    NullPolicy,
    compute_competence_vector,
    delayed_gratification_from_sortedness,
    detect_phase_boundaries,
    near_boundary_variance_table,
    parse_rule_program,
    phase_boundary_config_dict,
    run_smoke_checks,
    select_phase_boundary_candidates,
    summarize_repeated_seed_runs,
    validate_phase_boundary_outputs,
)
from morphospace.competence import COMPETENCE_VECTOR_VERSION, canonical_json, sortedness_sign_change_count  # noqa: E402
from morphospace.rule_dsl import DSL_VERSION  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S09"
STEP_NUMBER = 9
TITLE = "Find phase boundaries"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"

SEED_COUNT = 5
MAX_CANDIDATE_POLICIES = 420
SMOKE_POLICY_COUNT = 8
SMOKE_TASK_COUNT = 2
SMOKE_SEED_COUNT = 2

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_phase_boundaries"
FIGURES_DIR = STEP_DIR / "figures"

CORPUS_PATH = ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet"
S08_ELITES_PATH = RESULTS_DIR / "e03_qd_elites.parquet"
S08_STATUS_PATH = ARTIFACTS_DIR / "research_steps" / "S08" / "status.json"


PHASE_TASKS = (
    PhaseBoundaryTask(
        task_id="s09_unique_n4_boundary",
        task_family="sorting",
        task_panel="s09_cpu_phase_boundary",
        input_profile="unique_n4_boundary_probe",
        initial_values=(4, 1, 3, 2),
        scheduler_seed_base=41000,
        tie_seed_base=51000,
        max_activations=512,
        max_swaps=512,
        max_comparisons=2048,
    ),
    PhaseBoundaryTask(
        task_id="s09_duplicate_n5_boundary",
        task_family="sorting_duplicate_values",
        task_panel="s09_cpu_phase_boundary",
        input_profile="duplicate_n5_boundary_probe",
        initial_values=(2, 3, 1, 2, 1),
        scheduler_seed_base=42000,
        tie_seed_base=52000,
        max_activations=640,
        max_swaps=640,
        max_comparisons=2560,
    ),
    PhaseBoundaryTask(
        task_id="s09_stuck_frozen_n5_boundary",
        task_family="frozen",
        task_panel="s09_cpu_phase_boundary",
        input_profile="unique_n5_stuck_frozen_boundary_probe",
        initial_values=(5, 1, 4, 2, 3),
        scheduler_seed_base=43000,
        tie_seed_base=53000,
        frozen_variant="stuck",
        frozen_positions=(2,),
        max_activations=640,
        max_swaps=640,
        max_comparisons=2560,
    ),
    PhaseBoundaryTask(
        task_id="s09_chimera_aggregation_n6_boundary",
        task_family="chimera",
        task_panel="s09_cpu_phase_boundary",
        input_profile="alternating_dsl_null_chimera_n6_boundary_probe",
        initial_values=(6, 1, 5, 2, 4, 3),
        scheduler_seed_base=44000,
        tie_seed_base=54000,
        chimera_with_null=True,
        candidate_positions=(0, 2, 4),
        max_activations=768,
        max_swaps=768,
        max_comparisons=3072,
    ),
)


@dataclass(frozen=True)
class RunBundle:
    runs: pd.DataFrame
    vectors: pd.DataFrame


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def table_ready(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.columns:
        series = prepared[column]
        if series.map(lambda value: isinstance(value, (dict, list, tuple))).any():
            prepared[column] = series.map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple))
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


def value_counts_conserved(initial_values: tuple[int, ...], final_values: list[int]) -> bool:
    return Counter(map(int, initial_values)) == Counter(map(int, final_values))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [str(path) for path in [CORPUS_PATH, S08_ELITES_PATH, S08_STATUS_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"S09 requires completed S05/S08 inputs; missing: {missing}")
    s08_status = load_status(S08_STATUS_PATH)
    if not s08_status.get("success"):
        raise RuntimeError(f"S09 requires successful S08 status; got {s08_status}")
    return pd.read_parquet(CORPUS_PATH), pd.read_parquet(S08_ELITES_PATH)


def parser_validation_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        policy_id = str(row["policyId"])
        try:
            program = parse_rule_program(row["dslProgramJson"])
            roundtrip_ok = canonical_json(program.to_dict()) == canonical_json(parse_rule_program(program.to_json()).to_dict())
            rows.append(
                {
                    "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
                    "policyId": policy_id,
                    "parseSuccess": True,
                    "roundtripSuccess": bool(roundtrip_ok),
                    "dslVersion": program.version,
                    "ruleCountParsed": len(program.rules),
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - artifact path records unexpected parser failures.
            rows.append(
                {
                    "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
                    "policyId": policy_id,
                    "parseSuccess": False,
                    "roundtripSuccess": False,
                    "dslVersion": "",
                    "ruleCountParsed": 0,
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(rows)


def make_policies_for_task(program: Any, task: PhaseBoundaryTask):
    if not task.chimera_with_null:
        return DSLPolicy(program)
    candidate_positions = set(map(int, task.candidate_positions))
    policies = []
    for position in range(len(task.initial_values)):
        policies.append(DSLPolicy(program) if position in candidate_positions else NullPolicy())
    return policies


def trace_sortedness(result) -> list[float]:
    return [float(row["sortedness_percent"]) for row in result.trace_rows]


def trace_hashes(result) -> list[str]:
    return [str(row["state_hash"]) for row in result.trace_rows]


def run_one_policy_task_seed(row: pd.Series, task: PhaseBoundaryTask, seed_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_id = str(row["policyId"])
    program = parse_rule_program(row["dslProgramJson"])
    scheduler_seed = int(task.scheduler_seed_base + int(row["candidateRank"]) * 100 + int(seed_index))
    tie_seed = int(task.tie_seed_base + int(row["candidateRank"]) * 100 + int(seed_index))
    condition_id = f"S09::{task.task_id}::{policy_id}::seed{scheduler_seed}"
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
        implementation="e03_s09_cpu_phase_boundary",
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
    robustness_score = None if task.frozen_variant == "none" else float((1.0 if result.completed else 0.0) * final_sortedness_score)
    initial_algotype_aggregation = float(aggregation(result.initial_algotypes))
    final_algotype_aggregation = float(result.final_aggregation)

    run_record = {
        "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
        "researchStepId": STEP_ID,
        "policyId": policy_id,
        "candidateRank": int(row["candidateRank"]),
        "isS08Elite": bool(row["isS08Elite"]),
        "s08EliteRank": None if pd.isna(row.get("s08EliteRank")) else int(row["s08EliteRank"]),
        "familyKey": row["familyKey"],
        "parameterAxis": row["parameterAxis"],
        "parameterValue": float(row["parameterValue"]),
        "parameterLabel": row["parameterLabel"],
        "generationMethod": row["generationMethod"],
        "lineageId": row["lineageId"],
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
        "initialAggregation": initial_algotype_aggregation,
        "finalAggregation": final_algotype_aggregation,
        "robustnessScore": robustness_score,
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
        source_metric_source="s09_cpu_phase_boundary",
        source_artifact_path=str(STEP_DIR / "phase_boundary_repeated_seed_runs.parquet"),
    )
    vector.update(
        {
            "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
            "familyKey": row["familyKey"],
            "parameterAxis": row["parameterAxis"],
            "parameterValue": float(row["parameterValue"]),
            "candidateRank": int(row["candidateRank"]),
            "isS08Elite": bool(row["isS08Elite"]),
            "backend": "cpu_reference",
        }
    )
    return run_record, vector


def run_cpu_sweep(
    candidates: pd.DataFrame,
    tasks: tuple[PhaseBoundaryTask, ...],
    *,
    seed_count: int,
    policy_limit: int | None = None,
    task_limit: int | None = None,
) -> RunBundle:
    selected = candidates.sort_values("candidateRank", kind="mergesort")
    if policy_limit is not None:
        selected = selected.head(int(policy_limit))
    selected_tasks = tasks if task_limit is None else tasks[: int(task_limit)]
    run_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        for task in selected_tasks:
            for seed_index in range(int(seed_count)):
                run_record, vector = run_one_policy_task_seed(row, task, seed_index)
                run_rows.append(run_record)
                vector_rows.append(vector)
    return RunBundle(pd.DataFrame(run_rows), pd.DataFrame(vector_rows))


def write_phase_diagrams(boundaries_df: pd.DataFrame, summary_df: pd.DataFrame, variance_df: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if not boundaries_df.empty:
        top = boundaries_df.sort_values("boundaryScore", ascending=False, kind="mergesort").head(20).copy()
        labels = [f"{row.familyKey}\n{row.taskId}" for row in top.itertuples()]
        fig, ax = plt.subplots(figsize=(12, max(5, len(top) * 0.35)))
        ax.barh(np.arange(len(top)), top["boundaryScore"].astype(float), color="#2f6f7e")
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Boundary score")
        ax.set_title("Top S09 adjacent phase-boundary candidates")
        fig.tight_layout()
        path = FIGURES_DIR / "phase_boundary_scores_top20.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        top_pairs = top[["familyKey", "taskId"]].drop_duplicates().head(8)
        fig, axes = plt.subplots(len(top_pairs), 1, figsize=(12, max(4, len(top_pairs) * 2.1)), squeeze=False)
        for ax, (_, pair) in zip(axes[:, 0], top_pairs.iterrows()):
            family_key = pair["familyKey"]
            task_id = pair["taskId"]
            group = summary_df[summary_df["familyKey"].eq(family_key) & summary_df["taskId"].eq(task_id)].sort_values(
                ["parameterValue", "candidateRank"], kind="mergesort"
            )
            ax.plot(group["parameterValue"], group["meanFinalSortednessScore"], marker="o", label="final sortedness")
            ax.plot(group["parameterValue"], group["completionRate"], marker="s", label="completion")
            if group["meanOscillationProxy"].notna().any():
                ax.plot(group["parameterValue"], group["meanOscillationProxy"], marker="^", label="oscillation")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{family_key} / {task_id}", fontsize=9)
            ax.set_xlabel("Parameter value")
            ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        path = FIGURES_DIR / "phase_metric_traces_top_families.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    if not variance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            pd.to_numeric(variance_df["meanFinalSortednessScore"], errors="coerce"),
            pd.to_numeric(variance_df["finalSortednessScoreSd"], errors="coerce"),
            s=28,
            alpha=0.65,
            color="#9a4d2f",
        )
        ax.set_xlabel("Near-boundary mean final sortedness score")
        ax.set_ylabel("Repeated-seed SD")
        ax.set_title("Repeated-seed variance near S09 boundary candidates")
        fig.tight_layout()
        path = FIGURES_DIR / "near_boundary_seed_variance.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def copy_code_artifacts() -> list[Path]:
    paths: list[Path] = []
    for root in [STEP_DIR / "code", SHARED_CODE_DIR]:
        targets = [
            (REPO_ROOT / "scripts" / "e03_s09_phase_boundaries.py", root / "scripts" / "e03_s09_phase_boundaries.py"),
            (REPO_ROOT / "morphospace" / "phase_boundaries.py", root / "morphospace" / "phase_boundaries.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "morphospace" / "policies.py", root / "morphospace" / "policies.py"),
            (REPO_ROOT / "morphospace" / "rule_dsl.py", root / "morphospace" / "rule_dsl.py"),
            (REPO_ROOT / "tests" / "test_e03_phase_boundaries.py", root / "tests" / "test_e03_phase_boundaries.py"),
        ]
        for source, dest in targets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            paths.append(dest)
    return paths


def render_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    candidates_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    transition_counts: dict[str, int] = {}
    for column in [
        "sortingSuccessTransition",
        "sortednessTransition",
        "delayedGratificationTransition",
        "aggregationTransition",
        "oscillationTransition",
        "robustnessTransition",
    ]:
        if column in boundaries_df.columns:
            transition_counts[column] = int(boundaries_df[column].sum())
    top = boundaries_df.head(5)
    rows = [
        "# S09 Phase-Boundary Summary",
        "",
        "- Research step ID: S09",
        f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
        f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_phase_boundaries.parquet'}`, `{STEP_DIR / 'near_boundary_variance.parquet'}`, and `{FIGURES_DIR}/`",
        f"- Validation result: {validation_result}",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        (
            "- Lay summary: S09 replayed a bounded set of S08 elites and parameterized policy-family neighbors under repeated seeds. "
            "Adjacent policies in each family were compared to identify where small rule or probability changes flip sorting success, DG, aggregation, oscillation, or frozen-cell robustness."
        ),
        "",
        "## Key Counts",
        "",
        f"- Candidate policies: {len(candidates_df)}",
        f"- Repeated-seed CPU rows: {len(runs_df)}",
        f"- Phase-boundary candidates: {len(boundaries_df)}",
        f"- Near-boundary variance records: {len(variance_df)}",
        f"- Transition counts: `{canonical_json(transition_counts)}`",
    ]
    if len(top):
        rows.extend(["", "## Top Boundary Candidates", "", "| Rank | Family | Task | Score | Transitions |", "| --- | --- | --- | ---: | --- |"])
        for rank, row in enumerate(top.itertuples(), start=1):
            rows.append(
                f"| {rank} | `{row.familyKey}` | `{row.taskId}` | {float(row.boundaryScore):.3f} | `{row.transitionKindsJson}` |"
            )
    return "\n".join(rows) + "\n"


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[Path], caveats: list[str], recommended_next_action: str) -> str:
    rows = [
        "# S09 Validation Report",
        "",
        "- Research step ID: S09",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {int(validation_df['success'].sum())}/{len(validation_df)} checks passed",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for _, row in validation_df.iterrows():
        rows.append(f"| `{row['checkId']}` | {'pass' if row['success'] else 'fail'} | {str(row['detail']).replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    started_at = utc_now()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_written: list[Path] = []

    corpus_df, elites_df = load_inputs()
    candidates_df = select_phase_boundary_candidates(corpus_df, elites_df, max_policies=MAX_CANDIDATE_POLICIES)
    config = phase_boundary_config_dict(seed_count=SEED_COUNT, max_policies=MAX_CANDIDATE_POLICIES, tasks=PHASE_TASKS)

    config_path = STEP_DIR / "phase_boundary_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    parser_df = parser_validation_table(candidates_df)
    parser_failures = parser_df[~parser_df["parseSuccess"] | ~parser_df["roundtripSuccess"]]
    if len(parser_failures):
        raise RuntimeError(f"candidate DSL parser validation failed for {len(parser_failures)} policies")

    smoke_bundle = run_cpu_sweep(
        candidates_df,
        PHASE_TASKS,
        seed_count=SMOKE_SEED_COUNT,
        policy_limit=SMOKE_POLICY_COUNT,
        task_limit=SMOKE_TASK_COUNT,
    )
    smoke_checks_df = run_smoke_checks(smoke_bundle.runs, expected_min_rows=SMOKE_POLICY_COUNT * SMOKE_TASK_COUNT * SMOKE_SEED_COUNT)
    if not bool(smoke_checks_df["success"].all()):
        raise RuntimeError(f"S09 smoke checks failed: {smoke_checks_df.to_dict('records')}")

    full_bundle = run_cpu_sweep(candidates_df, PHASE_TASKS, seed_count=SEED_COUNT)
    summary_df = summarize_repeated_seed_runs(full_bundle.runs)
    boundaries_df = detect_phase_boundaries(summary_df)
    variance_df = near_boundary_variance_table(boundaries_df, summary_df)
    figure_paths = write_phase_diagrams(boundaries_df, summary_df, variance_df)

    artifacts_written.extend(write_table(candidates_df, STEP_DIR / "phase_boundary_candidate_policies.csv", STEP_DIR / "phase_boundary_candidate_policies.parquet"))
    artifacts_written.extend(write_table(parser_df, STEP_DIR / "candidate_parser_validation.csv", STEP_DIR / "candidate_parser_validation.parquet"))
    artifacts_written.extend(write_table(smoke_bundle.runs, STEP_DIR / "phase_boundary_smoke_runs.csv", STEP_DIR / "phase_boundary_smoke_runs.parquet"))
    artifacts_written.extend(write_table(smoke_checks_df, STEP_DIR / "phase_boundary_smoke_checks.csv", STEP_DIR / "phase_boundary_smoke_checks.parquet"))
    artifacts_written.extend(write_table(full_bundle.runs, STEP_DIR / "phase_boundary_repeated_seed_runs.csv", STEP_DIR / "phase_boundary_repeated_seed_runs.parquet"))
    artifacts_written.extend(write_table(full_bundle.runs, RESULTS_DIR / "e03_s09_phase_boundary_runs.csv", RESULTS_DIR / "e03_s09_phase_boundary_runs.parquet"))
    artifacts_written.extend(write_table(full_bundle.vectors, STEP_DIR / "phase_boundary_competence_vectors.csv", STEP_DIR / "phase_boundary_competence_vectors.parquet"))
    artifacts_written.extend(write_table(full_bundle.vectors, RESULTS_DIR / "e03_s09_phase_boundary_competence_vectors.csv", RESULTS_DIR / "e03_s09_phase_boundary_competence_vectors.parquet"))
    artifacts_written.extend(write_table(summary_df, STEP_DIR / "phase_boundary_policy_task_summary.csv", STEP_DIR / "phase_boundary_policy_task_summary.parquet"))
    artifacts_written.extend(write_table(boundaries_df, STEP_DIR / "phase_boundary_candidates.csv", STEP_DIR / "phase_boundary_candidates.parquet"))
    artifacts_written.extend(write_table(boundaries_df, RESULTS_DIR / "e03_phase_boundaries.csv", RESULTS_DIR / "e03_phase_boundaries.parquet"))
    artifacts_written.extend(write_table(variance_df, STEP_DIR / "near_boundary_variance.csv", STEP_DIR / "near_boundary_variance.parquet"))
    artifacts_written.extend(write_table(variance_df, RESULTS_DIR / "e03_s09_phase_boundary_variance.csv", RESULTS_DIR / "e03_s09_phase_boundary_variance.parquet"))
    artifacts_written.extend(figure_paths)

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_e03_phase_boundaries",
            "tests.test_e03_quality_diversity",
            "tests.test_e03_coarse_sweep",
            "tests.test_e03_batch_simulator",
            "tests.test_e03_competence",
        ],
        timeout=900,
    )
    repo_log = STEP_DIR / "repo_unit_test_log.txt"
    repo_log.write_text(repo_test_payload["output"], encoding="utf-8")
    artifacts_written.append(repo_log)

    upstream_statuses = {
        step: load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        for step in ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"]
    }
    validation_df = validate_phase_boundary_outputs(
        candidates=candidates_df,
        smoke_checks=smoke_checks_df,
        run_df=full_bundle.runs,
        summary_df=summary_df,
        boundaries_df=boundaries_df,
        variance_df=variance_df,
        expected_seed_count=SEED_COUNT,
        expected_task_count=len(PHASE_TASKS),
        upstream_statuses=upstream_statuses,
        repo_test_payload={key: value for key, value in repo_test_payload.items() if key != "output"},
        figure_paths=[str(path) for path in figure_paths if path.exists()],
        s10_dir_exists=(ARTIFACTS_DIR / "research_steps" / "S10").exists(),
    )
    parser_check = pd.DataFrame(
        [
            {
                "checkId": "candidate_dsl_parser_roundtrip",
                "success": bool(parser_df["parseSuccess"].all() and parser_df["roundtripSuccess"].all()),
                "detail": f"{len(parser_df)} candidate DSL records parsed and round-tripped",
            }
        ]
    )
    validation_df = pd.concat([parser_check, validation_df], ignore_index=True)
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "phase_boundary_validation.csv", STEP_DIR / "phase_boundary_validation.parquet"))
    artifacts_written.extend(write_table(validation_df, RESULTS_DIR / "e03_s09_phase_boundary_validation.csv", RESULTS_DIR / "e03_s09_phase_boundary_validation.parquet"))

    code_paths = copy_code_artifacts()
    artifacts_written.extend(code_paths)

    caveats = [
        "S09 uses small CPU-reference arrays and bounded candidate neighborhoods; it is phase-boundary screening, not a full large-array transfer claim.",
        "Aggregation is measured on an alternating DSL/null chimera probe using algotype labels, so it is a computational proxy for policy-family clustering.",
        "Repeated-seed variance reflects scheduler and stochastic-action variability over five seeds per policy/task, not a high-power confidence interval.",
        "Stateful, target, and stochastic DSL records run through the CPU interpreter; no new GPU kernel support is claimed.",
    ]
    recommended_next_action = (
        "Stop before S10 for Chief Scientist review; if approved, embed policy behavior using S09 boundary candidates, S08 elites, "
        "and the repeated-seed competence vectors as neighborhood anchors."
    )
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )

    summary_path = STEP_DIR / "summary.md"
    validation_report_path = STEP_DIR / "validation_report.md"
    summary_path.write_text(
        render_summary(
            validation_result=validation_result,
            artifacts_written=artifacts_written,
            candidates_df=candidates_df,
            runs_df=full_bundle.runs,
            boundaries_df=boundaries_df,
            variance_df=variance_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    validation_report_path.write_text(
        render_validation_report(validation_df, artifacts_written, caveats, recommended_next_action),
        encoding="utf-8",
    )
    artifacts_written.extend([summary_path, validation_report_path])

    boundary_kind_counts = {
        column: int(boundaries_df[column].sum())
        for column in [
            "sortingSuccessTransition",
            "sortednessTransition",
            "delayedGratificationTransition",
            "aggregationTransition",
            "oscillationTransition",
            "robustnessTransition",
        ]
        if column in boundaries_df.columns
    }
    status_path = STEP_DIR / "status.json"
    manifest_path = STEP_DIR / "artifact_manifest.json"
    git_metadata = get_git_metadata()
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
            "S09 used the S08 elite set and parameterized policy families to find adjacent policy changes where behavior flips under repeated seeds."
        ),
        "phaseBoundarySummary": {
            "candidatePolicyCount": int(len(candidates_df)),
            "taskCount": int(len(PHASE_TASKS)),
            "seedCountPerPolicyTask": int(SEED_COUNT),
            "runRowCount": int(len(full_bundle.runs)),
            "policyTaskSummaryRows": int(len(summary_df)),
            "boundaryCandidateCount": int(len(boundaries_df)),
            "nearBoundaryVarianceRows": int(len(variance_df)),
            "transitionKindCounts": boundary_kind_counts,
        },
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {
            "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
            "competenceVectorVersion": COMPETENCE_VECTOR_VERSION,
            "dslVersion": DSL_VERSION,
        },
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
        "manifestSelfReference": {
            "path": str(manifest_path),
            "sha256": "omitted_self_referential_manifest",
        },
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
                "candidatePolicyCount": int(len(candidates_df)),
                "runRowCount": int(len(full_bundle.runs)),
                "boundaryCandidateCount": int(len(boundaries_df)),
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
