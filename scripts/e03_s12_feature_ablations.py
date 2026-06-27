#!/usr/bin/env python3
"""Execute E03 S12: probe causal feature contributions with paired ablations.

S12 consumes completed S11 taxonomy artifacts and upstream S08-S10 policy
evidence.  It selects classic, generated, and elite DSL policies, creates
schema-valid ablations for local-rule primitives, replays originals and
ablations under identical CPU-reference seeds, computes competence deltas, and
stops before S13.
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
    ABLATION_TYPES,
    FEATURE_ABLATION_VERSION,
    AblationTask,
    DSLPolicy,
    NullPolicy,
    PolicyEventSimulator,
    build_ablation_policy_table,
    compute_competence_vector,
    feature_effect_summary,
    paired_delta_table,
    parse_rule_program,
    select_s12_source_policies,
    validate_ablation_outputs,
)
from morphospace.competence import canonical_json, delayed_gratification_from_sortedness, sortedness_sign_change_count  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S12"
STEP_NUMBER = 12
TITLE = "Probe causal features"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"

SEED_COUNT = 3
MAX_GENERATED_PER_CLASS = 3

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = STEP_DIR / "figures"
ABLATION_POLICY_DIR = STEP_DIR / "ablation_policies"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_feature_ablations"

CORPUS_PATH = ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet"
S08_ELITES_PATH = RESULTS_DIR / "e03_qd_elites.parquet"
S11_ASSIGNMENTS_PATH = RESULTS_DIR / "e03_s11_policy_class_assignments.parquet"
S11_CLASSES_PATH = RESULTS_DIR / "e03_universality_classes.parquet"
S11_EXEMPLARS_PATH = RESULTS_DIR / "e03_s11_class_exemplars.parquet"

S12_TASKS = (
    AblationTask(
        task_id="s12_unique_n5_same_seed",
        task_family="sorting",
        task_panel="s12_paired_ablation",
        input_profile="unique_n5_probe",
        initial_values=(5, 1, 4, 2, 3),
        scheduler_seed_base=61000,
        tie_seed_base=71000,
        max_activations=640,
        max_swaps=640,
        max_comparisons=2560,
    ),
    AblationTask(
        task_id="s12_duplicate_n5_same_seed",
        task_family="sorting_duplicate_values",
        task_panel="s12_paired_ablation",
        input_profile="duplicate_n5_probe",
        initial_values=(2, 3, 1, 2, 1),
        scheduler_seed_base=62000,
        tie_seed_base=72000,
        max_activations=640,
        max_swaps=640,
        max_comparisons=2560,
    ),
    AblationTask(
        task_id="s12_stuck_frozen_n5_same_seed",
        task_family="frozen",
        task_panel="s12_paired_ablation",
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
    AblationTask(
        task_id="s12_transfer_unique_n6_same_seed",
        task_family="transfer",
        task_panel="s12_paired_ablation",
        input_profile="unique_n6_transfer_probe",
        initial_values=(6, 1, 5, 2, 4, 3),
        scheduler_seed_base=64000,
        tie_seed_base=74000,
        max_activations=832,
        max_swaps=832,
        max_comparisons=3328,
    ),
    AblationTask(
        task_id="s12_chimera_null_n6_same_seed",
        task_family="chimera",
        task_panel="s12_paired_ablation",
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


def load_inputs() -> dict[str, pd.DataFrame]:
    required_paths = {
        "corpus": CORPUS_PATH,
        "s08_elites": S08_ELITES_PATH,
        "s11_assignments": S11_ASSIGNMENTS_PATH,
        "s11_classes": S11_CLASSES_PATH,
        "s11_exemplars": S11_EXEMPLARS_PATH,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"S12 requires completed S08-S11 inputs; missing: {missing}")
    for step in ["S08", "S09", "S10", "S11"]:
        status = load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        if not status.get("success"):
            raise RuntimeError(f"S12 requires successful {step}; got {status}")
    return {key: pd.read_parquet(path) for key, path in required_paths.items()}


def value_counts_conserved(initial_values: tuple[int, ...], final_values: list[int]) -> bool:
    return Counter(map(int, initial_values)) == Counter(map(int, final_values))


def trace_sortedness(result) -> list[float]:
    return [float(row["sortedness_percent"]) for row in result.trace_rows]


def trace_hashes(result) -> list[str]:
    return [str(row["state_hash"]) for row in result.trace_rows]


def write_ablation_policy_files(policy_table_df: pd.DataFrame) -> pd.DataFrame:
    ABLATION_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in policy_table_df.sort_values(["sourcePolicyId", "variantRole", "ablationType"], kind="mergesort").iterrows():
        policy_id = str(row["ablationPolicyId"])
        source_id = str(row["sourcePolicyId"])
        ablation_type = str(row["ablationType"])
        program = parse_rule_program(row["dslProgramJson"])
        path = ABLATION_POLICY_DIR / f"{source_id}__{ablation_type}__{policy_id}.json"
        path.write_text(json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "featureAblationVersion": FEATURE_ABLATION_VERSION,
                "sourcePolicyId": source_id,
                "ablationPolicyId": policy_id,
                "variantRole": row["variantRole"],
                "ablationType": ablation_type,
                "dslVersion": program.version,
                "dslPath": str(path),
                "dslSha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def parser_validation_table(policy_table_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in policy_table_df.iterrows():
        try:
            program = parse_rule_program(row["dslProgramJson"])
            roundtrip = canonical_json(parse_rule_program(program.to_json()).to_dict()) == canonical_json(program.to_dict())
            rows.append(
                {
                    "featureAblationVersion": FEATURE_ABLATION_VERSION,
                    "sourcePolicyId": row["sourcePolicyId"],
                    "ablationPolicyId": row["ablationPolicyId"],
                    "variantRole": row["variantRole"],
                    "ablationType": row["ablationType"],
                    "parseSuccess": True,
                    "roundtripSuccess": bool(roundtrip),
                    "ruleCountParsed": len(program.rules),
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - artifact path records unexpected parser failures.
            rows.append(
                {
                    "featureAblationVersion": FEATURE_ABLATION_VERSION,
                    "sourcePolicyId": row.get("sourcePolicyId", ""),
                    "ablationPolicyId": row.get("ablationPolicyId", ""),
                    "variantRole": row.get("variantRole", ""),
                    "ablationType": row.get("ablationType", ""),
                    "parseSuccess": False,
                    "roundtripSuccess": False,
                    "ruleCountParsed": 0,
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(rows)


def make_policies_for_task(program, task: AblationTask):
    if not task.chimera_with_null:
        return DSLPolicy(program)
    candidate_positions = set(map(int, task.candidate_positions))
    policies = []
    for position in range(len(task.initial_values)):
        policies.append(DSLPolicy(program) if position in candidate_positions else NullPolicy())
    return policies


def run_one_policy_task_seed(row: pd.Series, task: AblationTask, seed_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    source_rank = int(row["sourceRank"])
    policy_id = str(row["ablationPolicyId"])
    program = parse_rule_program(row["dslProgramJson"])
    scheduler_seed = int(task.scheduler_seed_base + source_rank * 100 + int(seed_index))
    tie_seed = int(task.tie_seed_base + source_rank * 100 + int(seed_index))
    condition_id = f"S12::{task.task_id}::{row['sourcePolicyId']}::{row['ablationType']}::seed{scheduler_seed}"
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
        implementation="e03_s12_cpu_paired_ablation",
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
        "featureAblationVersion": FEATURE_ABLATION_VERSION,
        "researchStepId": STEP_ID,
        "sourcePolicyId": str(row["sourcePolicyId"]),
        "ablationPolicyId": policy_id,
        "variantRole": str(row["variantRole"]),
        "ablationType": str(row["ablationType"]),
        "sourceRank": source_rank,
        "sourceRole": str(row["sourceRole"]),
        "sourceClassName": str(row["sourceClassName"]),
        "sourceClassLabel": str(row["sourceClassLabel"]),
        "sourceGenerationMethod": str(row["sourceGenerationMethod"]),
        "sourceLineageId": str(row["sourceLineageId"]),
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
        policy_family=str(row.get("sourceFamily", "unknown")),
        task_id=task.task_id,
        task_family=task.task_family,
        task_panel=task.task_panel,
        input_profile=task.input_profile,
        frozen_variant=task.frozen_variant,
        frozen_count=len(task.frozen_positions),
        replicate_index=seed_index,
        source_metric_source="s12_cpu_paired_ablation",
        source_artifact_path=str(STEP_DIR / "ablation_runs.parquet"),
        transfer_score=final_sortedness_score if task.task_family == "transfer" else None,
        compatibility_score=final_aggregation if task.task_family == "chimera" else None,
    )
    vector.update(
        {
            "featureAblationVersion": FEATURE_ABLATION_VERSION,
            "sourcePolicyId": str(row["sourcePolicyId"]),
            "ablationPolicyId": policy_id,
            "variantRole": str(row["variantRole"]),
            "ablationType": str(row["ablationType"]),
            "sourceRank": source_rank,
            "sourceRole": str(row["sourceRole"]),
            "sourceClassName": str(row["sourceClassName"]),
            "sourceClassLabel": str(row["sourceClassLabel"]),
            "sourceGenerationMethod": str(row["sourceGenerationMethod"]),
            "sourceLineageId": str(row["sourceLineageId"]),
            "schedulerSeed": scheduler_seed,
            "tieBreakerSeed": tie_seed,
            "backend": "cpu_reference",
        }
    )
    return run_record, vector


def run_paired_ablation_sweep(policy_table_df: pd.DataFrame, tasks: Sequence[AblationTask], *, seed_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for _, row in policy_table_df.sort_values(["sourceRank", "variantRole", "ablationType"], kind="mergesort").iterrows():
        for task in tasks:
            for seed_index in range(int(seed_count)):
                run_record, vector = run_one_policy_task_seed(row, task, seed_index)
                run_rows.append(run_record)
                vector_rows.append(vector)
    return pd.DataFrame(run_rows), pd.DataFrame(vector_rows)


def plot_outputs(paired_delta_df: pd.DataFrame, effect_summary_df: pd.DataFrame, source_policies_df: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if not paired_delta_df.empty:
        metric = "finalSortednessScoreDelta"
        summary = (
            paired_delta_df.groupby(["ablationType", "taskFamily"], dropna=False)[metric]
            .mean()
            .reset_index()
            .pivot(index="ablationType", columns="taskFamily", values=metric)
            .fillna(0.0)
        )
        fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.45 * len(summary))))
        image = ax.imshow(summary.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_yticks(np.arange(len(summary.index)))
        ax.set_yticklabels(summary.index, fontsize=8)
        ax.set_xticks(np.arange(len(summary.columns)))
        ax.set_xticklabels(summary.columns, rotation=35, ha="right", fontsize=8)
        ax.set_title("S12 mean final-sortedness delta by ablation and task family")
        fig.colorbar(image, ax=ax, label="ablated - original")
        fig.tight_layout()
        path = FIGURES_DIR / "s12_feature_delta_heatmap.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

        role_summary = paired_delta_df.groupby(["ablationType", "sourceRole"], dropna=False)[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5.8))
        labels = [f"{row.ablationType}\n{row.sourceRole}" for row in role_summary.itertuples()]
        x = np.arange(len(role_summary))
        colors = ["#2f6f7e" if value >= 0 else "#9a4d2f" for value in role_summary[metric]]
        ax.bar(x, role_summary[metric], color=colors)
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("mean final sortedness delta")
        ax.set_title("S12 paired ablation effects by source role")
        fig.tight_layout()
        path = FIGURES_DIR / "s12_feature_delta_by_role.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    role_counts = source_policies_df["sourceRole"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(role_counts.index, role_counts.values, color="#547c75")
    ax.set_ylabel("source policy count")
    ax.set_title("S12 source policy roles")
    fig.tight_layout()
    path = FIGURES_DIR / "s12_source_role_counts.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def render_causal_report(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    source_policies_df: pd.DataFrame,
    policy_table_df: pd.DataFrame,
    paired_delta_df: pd.DataFrame,
    effect_summary_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    top_loss = pd.DataFrame()
    if not effect_summary_df.empty and "finalSortednessScoreDeltaMean" in effect_summary_df.columns:
        top_loss = effect_summary_df.sort_values("finalSortednessScoreDeltaMean", kind="mergesort").head(12)
    role_counts = source_policies_df["sourceRole"].value_counts().to_dict()
    ablation_counts = policy_table_df["ablationType"].value_counts().to_dict()
    rows = [
        "# S12 Causal Feature Ablation Report",
        "",
        "- Research step ID: S12",
        f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
        f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_feature_ablations.parquet'}`, `{RESULTS_DIR / 'e03_s12_ablation_runs.parquet'}`, and `{FIGURES_DIR}/`",
        f"- Validation result: {validation_result}",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        (
            "- Lay summary: S12 removes or alters individual DSL features and compares each ablated policy with its original under the same simulator seeds. "
            "Large negative paired deltas mark candidate necessary contributors in this bounded panel; they do not prove biological or mathematical causality."
        ),
        "",
        "## Design",
        "",
        "- Source policies include parameterized classics, S08 elites, and generated S11 class exemplars.",
        "- Each ablated variant is a valid S02 DSL program with a checksummed JSON file.",
        "- Pairing keys are source policy, task, and replicate seed; originals and ablations share scheduler and tie-breaker seeds.",
        "- Tasks cover unique sorting, duplicate values, stuck Frozen Cell robustness, n=6 transfer, and DSL/null chimera aggregation.",
        "",
        "## Counts",
        "",
        f"- Source policies: {len(source_policies_df)} with roles `{canonical_json(role_counts)}`",
        f"- Policy variants: {len(policy_table_df)} with ablations `{canonical_json(ablation_counts)}`",
        f"- Paired deltas: {len(paired_delta_df)}",
        "",
        "## Candidate Feature Effects",
        "",
    ]
    if top_loss.empty:
        rows.append("- No effect summary rows were available.")
    else:
        rows.extend(
            [
                "| Ablation | Source Role | Task Family | Pairs | Mean Sortedness Delta | Mean Completion Delta | Call |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in top_loss.iterrows():
            rows.append(
                f"| `{row['ablationType']}` | `{row['sourceRole']}` | `{row['taskFamily']}` | {int(row['pairCount'])} | "
                f"{float(row.get('finalSortednessScoreDeltaMean', np.nan)):.3f} | "
                f"{float(row.get('completionSuccessDeltaMean', np.nan)):.3f} | `{row.get('candidateNecessityCall', '')}` |"
            )
    rows.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Candidate-necessary calls mean the feature removal hurt paired computational performance in this bounded task panel.",
            "- Removal-only ablations do not establish feature sufficiency; S12 records this explicitly for downstream S13/S14 interpretation.",
            "- Improvements after ablation are informative too: they identify costly or task-mismatched primitives in the current DSL policies.",
        ]
    )
    return "\n".join(rows) + "\n"


def render_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    source_policies_df: pd.DataFrame,
    policy_table_df: pd.DataFrame,
    run_df: pd.DataFrame,
    paired_delta_df: pd.DataFrame,
    effect_summary_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    source_roles = source_policies_df["sourceRole"].value_counts().to_dict()
    ablation_counts = policy_table_df["ablationType"].value_counts().to_dict()
    call_counts = effect_summary_df["candidateNecessityCall"].value_counts().to_dict() if "candidateNecessityCall" in effect_summary_df.columns else {}
    return "\n".join(
        [
            "# S12 Feature Ablation Summary",
            "",
            "- Research step ID: S12",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_feature_ablations.parquet'}`, `{STEP_DIR / 'causal_feature_report.md'}`, and `{FIGURES_DIR}/`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            (
                "- Lay summary: S12 tested which DSL primitives matter by comparing original policies with same-seed feature ablations across sorting, duplicate, frozen, transfer, and chimera tasks."
            ),
            "",
            "## Key Counts",
            "",
            f"- Source policies: {len(source_policies_df)} roles `{canonical_json(source_roles)}`",
            f"- Policy variants: {len(policy_table_df)} ablations `{canonical_json(ablation_counts)}`",
            f"- CPU run rows: {len(run_df)}",
            f"- Paired delta rows: {len(paired_delta_df)}",
            f"- Feature-effect summary calls: `{canonical_json(call_counts)}`",
        ]
    ) + "\n"


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[Path], caveats: list[str], recommended_next_action: str) -> str:
    rows = [
        "# S12 Validation Report",
        "",
        "- Research step ID: S12",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {int(validation_df['success'].sum())}/{len(validation_df)} checks passed",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        (
            "- Lay summary: S12 validation checks source-role coverage, parseable ablated DSL records, complete same-seed CPU runs, conserved values, paired deltas, reports, figures, tests, and no S13 artifacts."
        ),
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
            (REPO_ROOT / "scripts" / "e03_s12_feature_ablations.py", root / "scripts" / "e03_s12_feature_ablations.py"),
            (REPO_ROOT / "morphospace" / "feature_ablations.py", root / "morphospace" / "feature_ablations.py"),
            (REPO_ROOT / "morphospace" / "rule_dsl.py", root / "morphospace" / "rule_dsl.py"),
            (REPO_ROOT / "morphospace" / "policies.py", root / "morphospace" / "policies.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "tests" / "test_e03_feature_ablations.py", root / "tests" / "test_e03_feature_ablations.py"),
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
        "version": FEATURE_ABLATION_VERSION,
        "seedCount": SEED_COUNT,
        "maxGeneratedPerClass": MAX_GENERATED_PER_CLASS,
        "ablationTypes": list(ABLATION_TYPES),
        "tasks": [task.to_dict() for task in S12_TASKS],
        "inputs": {
            "corpus": str(CORPUS_PATH),
            "s08Elites": str(S08_ELITES_PATH),
            "s11Assignments": str(S11_ASSIGNMENTS_PATH),
            "s11Classes": str(S11_CLASSES_PATH),
            "s11Exemplars": str(S11_EXEMPLARS_PATH),
        },
    }
    config_path = STEP_DIR / "feature_ablation_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    source_policies_df = select_s12_source_policies(
        corpus_df=inputs["corpus"],
        elites_df=inputs["s08_elites"],
        assignments_df=inputs["s11_assignments"],
        exemplars_df=inputs["s11_exemplars"],
        max_generated_per_class=MAX_GENERATED_PER_CLASS,
    )
    source_policies_df["sourceRank"] = np.arange(len(source_policies_df), dtype=int)
    policy_table_df = build_ablation_policy_table(source_policies_df)
    policy_table_df = policy_table_df.merge(
        source_policies_df[["policyId", "sourceRank"]].rename(columns={"policyId": "sourcePolicyId"}),
        on="sourcePolicyId",
        how="left",
        validate="many_to_one",
    )
    policy_file_index_df = write_ablation_policy_files(policy_table_df)
    policy_table_df = policy_table_df.merge(
        policy_file_index_df[["ablationPolicyId", "dslPath", "dslSha256"]],
        on="ablationPolicyId",
        how="left",
        validate="one_to_one",
    )
    parser_df = parser_validation_table(policy_table_df)
    if not bool(parser_df["parseSuccess"].all() and parser_df["roundtripSuccess"].all()):
        raise RuntimeError("S12 parser validation failed before execution")

    run_df, vector_df = run_paired_ablation_sweep(policy_table_df, S12_TASKS, seed_count=SEED_COUNT)
    paired_df = paired_delta_table(vector_df, policy_table_df)
    effect_summary_df = feature_effect_summary(paired_df)

    artifacts_written.extend(write_table(source_policies_df, STEP_DIR / "source_policies.csv", STEP_DIR / "source_policies.parquet"))
    artifacts_written.extend(write_table(policy_table_df, STEP_DIR / "ablation_policy_specs.csv", STEP_DIR / "ablation_policy_specs.parquet"))
    artifacts_written.extend(
        write_table(policy_table_df, RESULTS_DIR / "e03_s12_ablation_policy_specs.csv", RESULTS_DIR / "e03_s12_ablation_policy_specs.parquet")
    )
    artifacts_written.extend(write_table(policy_file_index_df, STEP_DIR / "ablation_policy_file_index.csv", STEP_DIR / "ablation_policy_file_index.parquet"))
    artifacts_written.extend(write_table(parser_df, STEP_DIR / "ablation_parser_validation.csv", STEP_DIR / "ablation_parser_validation.parquet"))
    artifacts_written.extend(write_table(run_df, STEP_DIR / "ablation_runs.csv", STEP_DIR / "ablation_runs.parquet"))
    artifacts_written.extend(write_table(run_df, RESULTS_DIR / "e03_s12_ablation_runs.csv", RESULTS_DIR / "e03_s12_ablation_runs.parquet"))
    artifacts_written.extend(write_table(vector_df, STEP_DIR / "ablation_competence_vectors.csv", STEP_DIR / "ablation_competence_vectors.parquet"))
    artifacts_written.extend(
        write_table(
            vector_df,
            RESULTS_DIR / "e03_s12_ablation_competence_vectors.csv",
            RESULTS_DIR / "e03_s12_ablation_competence_vectors.parquet",
        )
    )
    artifacts_written.extend(write_table(paired_df, STEP_DIR / "feature_ablations.csv", STEP_DIR / "feature_ablations.parquet"))
    artifacts_written.extend(write_table(paired_df, RESULTS_DIR / "e03_feature_ablations.csv", RESULTS_DIR / "e03_feature_ablations.parquet"))
    artifacts_written.extend(write_table(effect_summary_df, STEP_DIR / "feature_effect_summary.csv", STEP_DIR / "feature_effect_summary.parquet"))
    artifacts_written.extend(
        write_table(effect_summary_df, RESULTS_DIR / "e03_s12_feature_effect_summary.csv", RESULTS_DIR / "e03_s12_feature_effect_summary.parquet")
    )

    figure_paths = plot_outputs(paired_df, effect_summary_df, source_policies_df)
    artifacts_written.extend(figure_paths)

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_e03_feature_ablations",
            "tests.test_e03_universality_classes",
            "tests.test_e03_behavior_embeddings",
            "tests.test_e03_quality_diversity",
            "tests.test_e03_competence",
        ],
        timeout=900,
    )
    repo_log = STEP_DIR / "repo_unit_test_log.txt"
    repo_log.write_text(repo_test_payload["output"], encoding="utf-8")
    artifacts_written.append(repo_log)

    caveats = [
        "S12 uses removal or alteration ablations, so it supports candidate necessity/contribution claims but not standalone sufficiency proofs.",
        "All measurements are bounded CPU-reference computational proxy results on small task panels, not biological causal evidence.",
        "Ablations in nonlinear rule programs can create interaction effects; a feature that hurts one policy class may help another.",
        "Chimera aggregation and transfer are screening tasks only and do not replace larger-array or full mixed-policy evaluations.",
    ]
    recommended_next_action = (
        "Stop before S13 for Chief Scientist review; if approved, compare classics against discovered policies using S11 classes and S12 ablation effects."
    )

    causal_report_path = STEP_DIR / "causal_feature_report.md"
    causal_report_path.write_text(
        render_causal_report(
            validation_result="pending until final validation table is written",
            artifacts_written=artifacts_written,
            source_policies_df=source_policies_df,
            policy_table_df=policy_table_df,
            paired_delta_df=paired_df,
            effect_summary_df=effect_summary_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    artifacts_written.append(causal_report_path)

    upstream_statuses = {
        step: load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json") for step in ["S08", "S09", "S10", "S11"]
    }
    validation_df = validate_ablation_outputs(
        source_policies_df=source_policies_df,
        policy_table_df=policy_table_df,
        run_df=run_df,
        vector_df=vector_df,
        paired_delta_df=paired_df,
        effect_summary_df=effect_summary_df,
        parser_validation_df=parser_df,
        upstream_statuses=upstream_statuses,
        repo_test_payload={key: value for key, value in repo_test_payload.items() if key != "output"},
        expected_task_count=len(S12_TASKS),
        seed_count=SEED_COUNT,
        report_exists=causal_report_path.exists(),
        figure_paths=[str(path) for path in figure_paths if path.exists()],
        s13_dir_exists=(ARTIFACTS_DIR / "research_steps" / "S13").exists(),
    )
    integrity_ok = bool(
        len(policy_file_index_df) == len(policy_table_df)
        and policy_file_index_df["dslPath"].map(lambda path: Path(str(path)).exists()).all()
        and policy_file_index_df.apply(lambda row: sha256_file(Path(str(row["dslPath"]))) == row["dslSha256"], axis=1).all()
    )
    validation_df = pd.concat(
        [
            validation_df,
            pd.DataFrame(
                [
                    {
                        "checkId": "ablation_policy_file_integrity",
                        "success": integrity_ok,
                        "detail": f"{len(policy_file_index_df)} checksummed ablation DSL files",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "feature_ablation_validation.csv", STEP_DIR / "feature_ablation_validation.parquet"))
    artifacts_written.extend(
        write_table(validation_df, RESULTS_DIR / "e03_s12_feature_ablation_validation.csv", RESULTS_DIR / "e03_s12_feature_ablation_validation.parquet")
    )

    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )
    causal_report_path.write_text(
        render_causal_report(
            validation_result=validation_result,
            artifacts_written=artifacts_written,
            source_policies_df=source_policies_df,
            policy_table_df=policy_table_df,
            paired_delta_df=paired_df,
            effect_summary_df=effect_summary_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
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
            source_policies_df=source_policies_df,
            policy_table_df=policy_table_df,
            run_df=run_df,
            paired_delta_df=paired_df,
            effect_summary_df=effect_summary_df,
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
    causal_report_path.write_text(
        render_causal_report(
            validation_result=validation_result,
            artifacts_written=final_artifact_paths,
            source_policies_df=source_policies_df,
            policy_table_df=policy_table_df,
            paired_delta_df=paired_df,
            effect_summary_df=effect_summary_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )

    git_metadata = get_git_metadata()
    source_roles = source_policies_df["sourceRole"].value_counts().to_dict()
    ablation_counts = policy_table_df["ablationType"].value_counts().to_dict()
    effect_calls = effect_summary_df["candidateNecessityCall"].value_counts().to_dict() if "candidateNecessityCall" in effect_summary_df.columns else {}
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
            "S12 compared original policies with same-seed DSL feature ablations across sorting, duplicate-value, frozen-cell, transfer, "
            "and chimera tasks to estimate bounded computational feature contributions."
        ),
        "featureAblationSummary": {
            "sourcePolicyCount": int(len(source_policies_df)),
            "sourceRoles": {str(key): int(value) for key, value in source_roles.items()},
            "policyVariantCount": int(len(policy_table_df)),
            "ablationCounts": {str(key): int(value) for key, value in ablation_counts.items()},
            "taskCount": int(len(S12_TASKS)),
            "seedCount": int(SEED_COUNT),
            "runRows": int(len(run_df)),
            "pairedDeltaRows": int(len(paired_df)),
            "effectSummaryRows": int(len(effect_summary_df)),
            "candidateNecessityCalls": {str(key): int(value) for key, value in effect_calls.items()},
        },
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {
            "featureAblationVersion": FEATURE_ABLATION_VERSION,
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
        "ablationPolicyFileCount": int(len(policy_file_index_df)),
        "ablationPolicyFileIndexPath": str(STEP_DIR / "ablation_policy_file_index.parquet"),
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
                "sourcePolicies": int(len(source_policies_df)),
                "policyVariants": int(len(policy_table_df)),
                "runRows": int(len(run_df)),
                "pairedDeltaRows": int(len(paired_df)),
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
