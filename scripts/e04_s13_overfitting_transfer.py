#!/usr/bin/env python3
"""Execute E04 S13 overfitting and transfer checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from memory_repair import (  # noqa: E402
    OVERFITTING_PROXY_SCOPE_NOTE,
    OVERFITTING_TRANSFER_VERSION,
    TRAINING_CONSTRAINT_VERSION,
    build_s13_policies,
    build_s13_tasks,
    competence_policy_from_s08_candidate,
    condition_rows_for_transfer_policy,
    run_transfer_condition,
    select_policies_from_selection_results,
    summarize_transfer_gaps,
    transfer_group_summary,
    validate_overfitting_outputs,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S13"
STEP_NUMBER = 13
STEP_TITLE = "Look for overfitting"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
S08_CANDIDATE_PATH = Path("/artifacts/results/e04_evolved_repair_policies.parquet")
TEST_MODULES = [
    "tests.test_e04_overfitting_transfer",
    "tests.test_e04_competence_proxies",
    "tests.test_e04_field_predictors",
    "tests.test_e04_gpu_evolution",
    "tests.test_e04_training_constraints",
    "tests.test_e04_local_learning",
    "tests.test_e04_homeostasis_tasks",
    "tests.test_e04_fatigue_damage",
    "tests.test_e04_repairable_frozen",
    "tests.test_e04_local_signals",
    "tests.test_e04_memory_extension",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def yaml_scalar(value: Any) -> str:
    value = json_ready(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#{}[]\n,") or text.lower() in {"true", "false", "null"}:
        return json.dumps(text)
    return text


def to_yaml(value: Any, indent: int = 0) -> str:
    value = json_ready(value)
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(value)}"


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(payload) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["success"] else "unknown",
        "branch": branch["stdout"].strip() if branch["success"] else "unknown",
        "remote": remote["stdout"].strip() if remote["success"] else "unknown",
        "statusShort": status["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def dataframe_to_artifacts(df: pd.DataFrame, step_path: Path, results_path: Path | None = None) -> list[Path]:
    df = df.copy()
    for column in df.columns:
        if df[column].map(lambda value: isinstance(value, (Mapping, list, tuple, np.ndarray))).any():
            df[column] = df[column].map(lambda value: compact_json(value) if isinstance(value, (Mapping, list, tuple, np.ndarray)) else value)
    step_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = step_path.with_suffix(".csv")
    parquet_path = step_path.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    written = [csv_path, parquet_path]
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_csv = results_path.with_suffix(".csv")
        results_parquet = results_path.with_suffix(".parquet")
        shutil.copy2(csv_path, results_csv)
        shutil.copy2(parquet_path, results_parquet)
        written.extend([results_csv, results_parquet])
    return written


def load_evolved_rows(max_evolved: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not S08_CANDIDATE_PATH.exists():
        return [], [{"familyKind": "evolved", "replayable": False, "replayBlocker": f"missing {S08_CANDIDATE_PATH}"}]
    df = pd.read_parquet(S08_CANDIDATE_PATH).sort_values("rank", kind="mergesort")
    rows = []
    catalog_rows = []
    for _, row in df.iterrows():
        policy = competence_policy_from_s08_candidate(row)
        catalog_rows.append(policy.to_dict())
        if policy.replayable and len(rows) < max_evolved:
            rows.append(row.to_dict())
    return rows, catalog_rows


def plot_transfer_results(gap_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if gap_df.empty:
        return []
    ordered = gap_df.sort_values("selectionMeanScore", ascending=False, kind="mergesort").copy()
    labels = [str(policy).replace("enhanced_", "enh_").replace("baseline_", "base_") for policy in ordered["policyId"]]
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.38
    ax.bar(x - width / 2, ordered["selectionMeanScore"], width=width, label="selection", color="#4776A6")
    ax.bar(x + width / 2, ordered["holdoutMeanScore"], width=width, label="holdout", color="#A66B45")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean computational transfer score")
    ax.set_title("S13 selection versus holdout policy scores")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    score_png = figures_dir / "e04_s13_selection_vs_holdout_scores.png"
    score_svg = figures_dir / "e04_s13_selection_vs_holdout_scores.svg"
    fig.savefig(score_png, dpi=160)
    fig.savefig(score_svg)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#4B8F6A" if selected else "#777777" for selected in ordered["selectedForHoldoutReview"]]
    ax.bar(labels, ordered["holdoutMinusSelectionScore"], color=colors)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Holdout minus selection score")
    ax.set_title("S13 generalization gap proxy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    gap_png = figures_dir / "e04_s13_generalization_gaps.png"
    gap_svg = figures_dir / "e04_s13_generalization_gaps.svg"
    fig.savefig(gap_png, dpi=160)
    fig.savefig(gap_svg)
    plt.close(fig)
    return [score_png, score_svg, gap_png, gap_svg]


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_root = step_dir / "code"
    sources = [
        REPO_ROOT / "memory_repair" / "overfitting_transfer.py",
        REPO_ROOT / "memory_repair" / "competence_proxies.py",
        REPO_ROOT / "memory_repair" / "memory.py",
        REPO_ROOT / "memory_repair" / "signals.py",
        REPO_ROOT / "memory_repair" / "repair.py",
        REPO_ROOT / "memory_repair" / "fatigue.py",
        REPO_ROOT / "memory_repair" / "homeostasis.py",
        REPO_ROOT / "memory_repair" / "learning.py",
        REPO_ROOT / "memory_repair" / "training_constraints.py",
        REPO_ROOT / "scripts" / "e04_s13_overfitting_transfer.py",
        REPO_ROOT / "tests" / "test_e04_overfitting_transfer.py",
    ]
    written = []
    for source in sources:
        destination = code_root / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        written.append(destination)
    return written


def artifact_manifest(artifacts_dir: Path, artifacts: Sequence[Path]) -> dict[str, Any]:
    records = []
    for path in sorted(set(artifacts), key=lambda item: str(item)):
        if not path.exists() or path.is_dir():
            continue
        records.append(
            {
                "path": str(path),
                "relativeToArtifactsDir": str(path.relative_to(artifacts_dir)) if path.is_relative_to(artifacts_dir) else str(path),
                "sizeBytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
            }
        )
    return {
        "schema": "eidosoma.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "artifactCount": len(records),
        "createdAt": utc_now(),
        "artifacts": records,
    }


def write_reports(
    *,
    step_dir: Path,
    task_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    group_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    outcome: str,
    caveats: Sequence[str],
    recommended_next_action: str,
) -> list[Path]:
    reports = []
    validation_report = step_dir / "validation_report.md"
    validation_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Validation Report",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: `{step_dir / 'overfitting_transfer_validation.csv'}` and `.parquet`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                markdown_table(["checkId", "success", "detail"], [[row.checkId, row.success, row.detail] for row in validation_df.itertuples(index=False)]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(validation_report)

    split_report = step_dir / "selection_holdout_split_report.md"
    split_rows = [
        [
            row.taskId,
            row.splitRole,
            row.initialLength,
            row.perturbationRegime,
            row.frozenPlacementCategory,
            row.recoveryNudgeThreshold,
            row.recoverySignalThreshold,
            row.noiseRegime,
        ]
        for row in task_df.itertuples(index=False)
    ]
    split_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Selection-Holdout Split Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'overfitting_transfer_tasks.csv'}`, `{step_dir / 'overfitting_transfer_conditions.csv'}`, and validation outputs",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                OVERFITTING_PROXY_SCOPE_NOTE,
                "",
                "The policy selection rule uses only rows marked `splitRole=selection`. Holdout rows have disjoint sizes, perturbation signatures, frozen-placement signatures, recovery thresholds, noise regimes, and seeds.",
                "",
                markdown_table(
                    ["taskId", "split", "n", "perturbation", "frozen placement", "nudge", "signal", "noise"],
                    split_rows,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(split_report)

    report = step_dir / "overfitting_transfer_report.md"
    report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Overfitting Transfer Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'overfitting_transfer_results.csv'}`, `{step_dir / 'overfitting_policy_gaps.csv'}`, `{step_dir / 'overfitting_selection_decisions.csv'}`, and S13 figures",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                OVERFITTING_PROXY_SCOPE_NOTE,
                "",
                "Selection-only decisions:",
                "",
                markdown_table(
                    ["policyId", "group", "selection rank", "selection score", "rule", "holdout used"],
                    [
                        [
                            row.selectedPolicyId,
                            row.policyGroup,
                            row.selectionRankWithinGroup,
                            row.selectionOnlyMeanScore,
                            row.decisionRule,
                            row.holdoutResultsUsedForSelection,
                        ]
                        for row in decision_df.itertuples(index=False)
                    ],
                ),
                "",
                "Policy-level transfer gaps:",
                "",
                markdown_table(
                    ["policyId", "group", "selected", "selection", "holdout", "holdout-selection", "rank shift"],
                    [
                        [
                            row.policyId,
                            row.policyGroup,
                            row.selectedForHoldoutReview,
                            row.selectionMeanScore,
                            row.holdoutMeanScore,
                            row.holdoutMinusSelectionScore,
                            row.rankShiftHoldoutMinusSelection,
                        ]
                        for row in gap_df.itertuples(index=False)
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(report)

    selected = gap_df[gap_df["selectedForHoldoutReview"].astype(bool)] if not gap_df.empty else gap_df
    mean_gap = None if selected.empty else float(selected["holdoutMinusSelectionScore"].mean())
    mean_drop = None if selected.empty else float(selected["relativeSelectionToHoldoutDrop"].mean())
    summary = step_dir / "summary.md"
    summary.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Status Summary",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: primary outputs under `{step_dir}` plus shared results/config/figure copies; full list is in `status.json` and `artifact_manifest.json`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "Lay summary: S13 selected policies using only a mid-size, lower-noise computational panel, then measured those same policies on larger arrays with different perturbation schedules, frozen placements, recovery thresholds, noise levels, and seeds. The reported gaps are direct computational transfer proxies.",
                "",
                OVERFITTING_PROXY_SCOPE_NOTE,
                "",
                f"Outcome classification: {outcome}.",
                f"Task rows: {len(task_df)}; condition rows: {len(condition_df)}; selected policies: {len(decision_df)}.",
                f"Mean selected holdout-minus-selection score: `{mean_gap}`.",
                f"Mean selected relative drop: `{mean_drop}`.",
                "",
                "Group summary:",
                "",
                markdown_table(
                    ["group", "policies", "selected", "selection", "holdout", "holdout-selection", "relative drop"],
                    [
                        [
                            row.policyGroup,
                            row.policyCount,
                            row.selectedPolicyCount,
                            row.meanSelectionScore,
                            row.meanHoldoutScore,
                            row.meanHoldoutMinusSelectionScore,
                            row.meanRelativeDrop,
                        ]
                        for row in group_df.itertuples(index=False)
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(summary)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--max-evolved", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=16100)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    figures_dir = artifacts_dir / "figures"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    artifacts: list[Path] = []

    evolved_rows, evolved_catalog = load_evolved_rows(args.max_evolved)
    tasks = build_s13_tasks()
    policies = build_s13_policies(evolved_rows, max_evolved=args.max_evolved)
    policy_catalog = [policy.to_dict() for policy in policies] + [
        {**row, "catalogOnly": True} for row in evolved_catalog if not row.get("replayable", False)
    ]
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "seedCount": int(args.seed_count),
        "seedBase": int(args.seed_base),
        "maxEvolved": int(args.max_evolved),
        "serialWorkerCount": 1,
        "s08CandidatePath": str(S08_CANDIDATE_PATH),
        "splitRoles": ["selection", "holdout"],
        "selectionRule": "select top baseline policy and top two enhanced policies by mean selection-panel conditionTransferScore",
        "holdoutSeparationContract": [
            "sizes",
            "perturbation signatures",
            "frozen placement signatures",
            "recovery thresholds",
            "noise regimes",
            "scheduler/tie/signal seeds",
        ],
        "proxyScopeNote": OVERFITTING_PROXY_SCOPE_NOTE,
        "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "createdAt": utc_now(),
    }
    for path in [step_dir / "overfitting_transfer_config.json", configs_dir / "e04_s13_overfitting_transfer.json"]:
        write_json(path, config_payload)
        artifacts.append(path)
    for path in [step_dir / "overfitting_transfer_config.yaml", configs_dir / "e04_s13_overfitting_transfer.yaml"]:
        write_yaml(path, config_payload)
        artifacts.append(path)

    task_df = pd.DataFrame([task.to_dict() for task in tasks])
    policy_df = pd.DataFrame(policy_catalog)
    artifacts.extend(dataframe_to_artifacts(task_df, step_dir / "overfitting_transfer_tasks", results_dir / "e04_s13_overfitting_transfer_tasks"))
    artifacts.extend(dataframe_to_artifacts(policy_df, step_dir / "overfitting_transfer_policies", results_dir / "e04_s13_overfitting_transfer_policies"))

    conditions = []
    for policy_index, policy in enumerate(policies):
        conditions.extend(
            condition_rows_for_transfer_policy(
                policy,
                tasks,
                seed_count=int(args.seed_count),
                seed_base=int(args.seed_base + policy_index * 10_000),
            )
        )
    condition_df = pd.DataFrame(conditions)
    artifacts.extend(dataframe_to_artifacts(condition_df, step_dir / "overfitting_transfer_conditions", results_dir / "e04_s13_overfitting_transfer_conditions"))

    policy_by_id = {policy.policy_id: policy for policy in policies}
    result_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    for condition in conditions:
        result, ticks = run_transfer_condition(policy_by_id[str(condition["policyId"])], condition)
        result_rows.append(result)
        tick_rows.extend(ticks)
    result_df = pd.DataFrame(result_rows)
    tick_df = pd.DataFrame(tick_rows)
    decision_df = select_policies_from_selection_results(result_df)
    gap_df = summarize_transfer_gaps(result_df, decision_df)
    group_df = transfer_group_summary(gap_df)
    validation_df = validate_overfitting_outputs(task_df, condition_df, result_df, decision_df, gap_df)

    artifacts.extend(dataframe_to_artifacts(result_df, step_dir / "overfitting_transfer_results", results_dir / "e04_overfitting_transfer"))
    artifacts.extend(dataframe_to_artifacts(tick_df, step_dir / "overfitting_transfer_tick_records", results_dir / "e04_s13_overfitting_transfer_tick_records"))
    artifacts.extend(dataframe_to_artifacts(decision_df, step_dir / "overfitting_selection_decisions", results_dir / "e04_s13_overfitting_selection_decisions"))
    artifacts.extend(dataframe_to_artifacts(gap_df, step_dir / "overfitting_policy_gaps", results_dir / "e04_s13_overfitting_policy_gaps"))
    artifacts.extend(dataframe_to_artifacts(group_df, step_dir / "overfitting_group_summary", results_dir / "e04_s13_overfitting_group_summary"))
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "overfitting_transfer_validation", results_dir / "e04_s13_overfitting_transfer_validation"))

    figure_paths = plot_transfer_results(gap_df, figures_dir)
    for figure in figure_paths:
        step_figure = step_dir / figure.name
        shutil.copy2(figure, step_figure)
        artifacts.extend([figure, step_figure])

    test_result = {"args": ["skipped"], "returncode": 0, "success": True, "stdout": "", "stderr": "", "runtimeSeconds": 0.0}
    if not args.skip_tests:
        test_result = run_command([sys.executable, "-m", "unittest", *TEST_MODULES], cwd=REPO_ROOT)
    test_log = step_dir / "repo_unit_test_log.txt"
    test_log.write_text(
        "\n".join(
            [
                f"command: {' '.join(test_result['args'])}",
                f"returncode: {test_result['returncode']}",
                f"runtimeSeconds: {test_result['runtimeSeconds']}",
                "",
                "STDOUT:",
                test_result["stdout"],
                "",
                "STDERR:",
                test_result["stderr"],
            ]
        ),
        encoding="utf-8",
    )
    artifacts.append(test_log)

    validation_df = pd.concat(
        [
            validation_df,
            pd.DataFrame(
                [
                    {
                        "checkId": "relevant_unit_tests_passed",
                        "success": bool(test_result["success"]),
                        "detail": f"Command {' '.join(test_result['args'])} returned {test_result['returncode']}",
                        "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
                        "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "overfitting_transfer_validation", results_dir / "e04_s13_overfitting_transfer_validation"))

    code_artifacts = copy_code_artifacts(step_dir)
    artifacts.extend(code_artifacts)

    selected = gap_df[gap_df["selectedForHoldoutReview"].astype(bool)] if not gap_df.empty else gap_df
    mean_selected_gap = None if selected.empty else float(selected["holdoutMinusSelectionScore"].mean())
    mean_selected_drop = None if selected.empty else float(selected["relativeSelectionToHoldoutDrop"].mean())
    if mean_selected_gap is None or pd.isna(mean_selected_gap):
        outcome = "null"
    elif mean_selected_gap < -0.05:
        outcome = "constraining/contradictory"
    elif abs(mean_selected_gap) <= 0.02:
        outcome = "null"
    else:
        outcome = "supportive"
    caveats = [
        "S13 scores are direct computational transfer proxies and do not measure biological robustness or learning.",
        "Selection and holdout panels are deliberately small stress tests; they characterize overfitting risk rather than exhaustive generalization.",
        "Selection policies were chosen by mean selection-panel score only; holdout results are reserved for post-selection evaluation.",
        "Selection and holdout tasks differ in several axes at once, so gaps localize transfer risk but do not isolate a single causal stressor.",
    ]
    recommended_next_action = "Chief review of S13 overfitting and transfer evidence; do not start S14 from this run."
    reports = write_reports(
        step_dir=step_dir,
        task_df=task_df,
        condition_df=condition_df,
        decision_df=decision_df,
        gap_df=gap_df,
        group_df=group_df,
        validation_df=validation_df,
        outcome=outcome,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
    )
    artifacts.extend(reports)

    success = bool(validation_df["success"].all())
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    artifacts.extend([status_path, manifest_path])
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_failed_validation",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": {
            "passedCount": int(validation_df["success"].sum()),
            "totalCount": int(len(validation_df)),
            "allPassed": success,
            "validationResultsPath": str(step_dir / "overfitting_transfer_validation.csv"),
        },
        "caveatsOrBlockers": caveats if success else caveats + ["At least one S13 validation check failed."],
        "recommendedNextAction": recommended_next_action,
        "experimentId": EXPERIMENT_ID,
        "title": STEP_TITLE,
        "outcomeClassification": outcome,
        "startedAt": config_payload["createdAt"],
        "completedAt": utc_now(),
        "runtimeSeconds": time.perf_counter() - started,
        "git": get_git_metadata(),
        "platform": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor()},
        "taskCount": int(len(task_df)),
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(result_df)),
        "selectedPolicyCount": int(len(decision_df)),
        "policyGapRowCount": int(len(gap_df)),
        "meanSelectedHoldoutMinusSelectionScore": mean_selected_gap,
        "meanSelectedRelativeDrop": mean_selected_drop,
        "selectionInitialLengths": sorted(int(value) for value in condition_df.loc[condition_df["splitRole"] == "selection", "initialLength"].unique()),
        "holdoutInitialLengths": sorted(int(value) for value in condition_df.loc[condition_df["splitRole"] == "holdout", "initialLength"].unique()),
        "serialWorkerCount": 1,
        "proxyScopeNote": OVERFITTING_PROXY_SCOPE_NOTE,
        "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    write_json(status_path, status)
    manifest = artifact_manifest(artifacts_dir, [path for path in artifacts if path != manifest_path])
    write_json(manifest_path, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
