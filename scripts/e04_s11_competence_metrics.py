#!/usr/bin/env python3
"""Execute E04 S11 held-out competence-proxy measurements."""

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
    COMPETENCE_AXES,
    COMPETENCE_PROXY_VERSION,
    PROXY_SCOPE_NOTE,
    baseline_competence_policies,
    build_s11_tasks,
    competence_group_comparison,
    competence_metric_definitions,
    competence_metric_table,
    competence_policy_from_s08_candidate,
    condition_rows_for_competence_policy,
    local_memory_signal_competence_policy,
    run_competence_condition,
    summarize_competence_profiles,
    validate_competence_outputs,
)
from memory_repair.training_constraints import TRAINING_CONSTRAINT_VERSION  # noqa: E402


EXPERIMENT_ID = "E04"
STEP_ID = "S11"
STEP_NUMBER = 11
STEP_TITLE = "Measure intelligence-like competence proxies"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
S08_CANDIDATE_PATH = Path("/artifacts/results/e04_evolved_repair_policies.parquet")
TEST_MODULES = [
    "tests.test_e04_competence_proxies",
    "tests.test_e04_communication_ablations",
    "tests.test_e04_memory_ablations",
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


def load_evolved_policies(max_evolved: int) -> tuple[list[Any], list[dict[str, Any]]]:
    if not S08_CANDIDATE_PATH.exists():
        return [], [{"familyKind": "evolved", "replayable": False, "replayBlocker": f"missing {S08_CANDIDATE_PATH}"}]
    df = pd.read_parquet(S08_CANDIDATE_PATH).sort_values("rank", kind="mergesort")
    policies = []
    catalog_rows = []
    for _, row in df.iterrows():
        policy = competence_policy_from_s08_candidate(row)
        catalog_rows.append(policy.to_dict())
        if policy.replayable and len(policies) < max_evolved:
            policies.append(policy)
    return policies, catalog_rows


def plot_competence_radar(profile_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    axes = [
        ("goalAttainmentProxyMean", "Goal"),
        ("multipleRoutesToGoalProxy", "Routes"),
        ("barrierCircumventionProxyMean", "Barrier"),
        ("recoveryAfterPerturbationProxyMean", "Recovery"),
        ("transferToLargerArraysProxyMean", "Transfer"),
        ("gracefulDegradationProxy", "Graceful"),
    ]
    if profile_df.empty:
        return []
    grouped = profile_df.groupby("policyGroup", dropna=False)[[column for column, _ in axes]].mean()
    labels = [label for _, label in axes]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    colors = {"baseline": "#4776A6", "enhanced": "#4B8F6A"}
    for group, values in grouped.iterrows():
        scores = [float(values[column]) if pd.notna(values[column]) else 0.0 for column, _ in axes]
        scores += scores[:1]
        ax.plot(angles, scores, label=str(group), linewidth=2.0, color=colors.get(str(group), "#7B6BAF"))
        ax.fill(angles, scores, alpha=0.12, color=colors.get(str(group), "#7B6BAF"))
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1], labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("S11 held-out computational competence proxies")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    fig.tight_layout()
    png_path = figures_dir / "e04_s11_competence_proxy_radar.png"
    svg_path = figures_dir / "e04_s11_competence_proxy_radar.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    return [png_path, svg_path]


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_root = step_dir / "code"
    sources = [
        REPO_ROOT / "memory_repair" / "competence_proxies.py",
        REPO_ROOT / "memory_repair" / "memory.py",
        REPO_ROOT / "memory_repair" / "signals.py",
        REPO_ROOT / "memory_repair" / "repair.py",
        REPO_ROOT / "memory_repair" / "fatigue.py",
        REPO_ROOT / "memory_repair" / "homeostasis.py",
        REPO_ROOT / "memory_repair" / "learning.py",
        REPO_ROOT / "memory_repair" / "training_constraints.py",
        REPO_ROOT / "scripts" / "e04_s11_competence_metrics.py",
        REPO_ROOT / "tests" / "test_e04_competence_proxies.py",
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


def write_metric_definition_markdown(path: Path, definition_df: pd.DataFrame) -> None:
    rows = [
        [row.metricId, row.aggregationLevel, row.directComputationalMetric, row.normalization]
        for row in definition_df.itertuples(index=False)
    ]
    path.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Competence Proxy Metric Definitions",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{path}` plus `competence_proxy_metric_definitions.csv` and `.parquet`",
                "- Validation result: definitions are direct computational proxy metrics with explicit normalization and claim boundaries",
                "- Caveats or blockers: none at definition time; empirical caveats are in `summary.md` and `validation_report.md`",
                "- Recommended next action: Chief review of S11 before any S12 work",
                "",
                PROXY_SCOPE_NOTE,
                "",
                markdown_table(["metricId", "level", "direct computational metric", "normalization"], rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_reports(
    *,
    step_dir: Path,
    result_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
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
                f"- Artifacts written: `{step_dir / 'competence_proxy_validation.csv'}` and `.parquet`",
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

    report = step_dir / "competence_proxy_report.md"
    profile_preview = profile_df.sort_values("overallCompetenceProxy", ascending=False, kind="mergesort").head(12)
    comparison_preview = comparison_df.head(12)
    report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Competence Proxy Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'competence_proxy_results.csv'}`, `{step_dir / 'competence_policy_profiles.csv'}`, `{step_dir / 'competence_metrics.csv'}`, and radar figures",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                PROXY_SCOPE_NOTE,
                "",
                "Policy profile preview:",
                "",
                markdown_table(
                    ["policyId", "group", "overall", "goal", "routes", "barrier", "recovery", "transfer", "graceful"],
                    [
                        [
                            row.policyId,
                            row.policyGroup,
                            row.overallCompetenceProxy,
                            row.goalAttainmentProxyMean,
                            row.multipleRoutesToGoalProxy,
                            row.barrierCircumventionProxyMean,
                            row.recoveryAfterPerturbationProxyMean,
                            row.transferToLargerArraysProxyMean,
                            row.gracefulDegradationProxy,
                        ]
                        for row in profile_preview.itertuples(index=False)
                    ],
                ),
                "",
                "Baseline versus enhanced group means:",
                "",
                markdown_table(
                    ["metric", "baselineMean", "enhancedMean", "enhancedMinusBaseline"],
                    [
                        [row.metricId, row.baselineMean, row.enhancedMean, row.enhancedMinusBaseline]
                        for row in comparison_preview.itertuples(index=False)
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(report)

    overall_row = comparison_df[comparison_df["metricId"] == "overallCompetenceProxy"]
    overall_delta = None if overall_row.empty else overall_row["enhancedMinusBaseline"].iloc[0]
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
                "Lay summary: S11 measured held-out computational proxy scores for baseline and enhanced local policies. The scores cover goal attainment, multiple route signatures to the same sorted goal, repairable-barrier circumvention, recovery after new perturbation schedules, transfer to larger arrays, and graceful degradation under stronger damage/fatigue.",
                "",
                PROXY_SCOPE_NOTE,
                "",
                f"Outcome classification: {outcome}.",
                f"Condition rows: {len(result_df)}; policy profiles: {len(profile_df)}.",
                f"Enhanced minus baseline overall competence proxy: `{overall_delta}`.",
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
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--max-evolved", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=12100)
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

    tasks = build_s11_tasks()
    evolved_policies, evolved_catalog = load_evolved_policies(args.max_evolved)
    policies = list(baseline_competence_policies()) + [local_memory_signal_competence_policy()] + evolved_policies
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
        "proxyScopeNote": PROXY_SCOPE_NOTE,
        "competenceAxes": list(COMPETENCE_AXES),
        "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "createdAt": utc_now(),
    }
    for path in [step_dir / "competence_proxy_config.json", configs_dir / "e04_s11_competence_metrics.json"]:
        write_json(path, config_payload)
        artifacts.append(path)
    for path in [step_dir / "competence_proxy_config.yaml", configs_dir / "e04_s11_competence_metrics.yaml"]:
        write_yaml(path, config_payload)
        artifacts.append(path)

    definition_df = pd.DataFrame(competence_metric_definitions())
    task_df = pd.DataFrame([task.to_dict() for task in tasks])
    policy_df = pd.DataFrame(policy_catalog)
    artifacts.extend(dataframe_to_artifacts(definition_df, step_dir / "competence_proxy_metric_definitions", results_dir / "e04_s11_competence_proxy_metric_definitions"))
    definition_md = step_dir / "competence_proxy_metric_definitions.md"
    write_metric_definition_markdown(definition_md, definition_df)
    artifacts.append(definition_md)
    artifacts.extend(dataframe_to_artifacts(task_df, step_dir / "competence_proxy_tasks", results_dir / "e04_s11_competence_proxy_tasks"))
    artifacts.extend(dataframe_to_artifacts(policy_df, step_dir / "competence_proxy_policies", results_dir / "e04_s11_competence_proxy_policies"))

    conditions = []
    for policy_index, policy in enumerate(policies):
        conditions.extend(
            condition_rows_for_competence_policy(
                policy,
                tasks,
                seed_count=int(args.seed_count),
                seed_base=int(args.seed_base + policy_index * 1000),
            )
        )
    condition_df = pd.DataFrame(conditions)
    artifacts.extend(dataframe_to_artifacts(condition_df, step_dir / "competence_proxy_conditions", results_dir / "e04_s11_competence_proxy_conditions"))

    policy_by_id = {policy.policy_id: policy for policy in policies}
    result_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    for condition in conditions:
        result, ticks = run_competence_condition(policy_by_id[str(condition["policyId"])], condition)
        result_rows.append(result)
        tick_rows.extend(ticks)
    result_df = pd.DataFrame(result_rows)
    tick_df = pd.DataFrame(tick_rows)
    profile_df = summarize_competence_profiles(result_df)
    metric_df = competence_metric_table(result_df, profile_df)
    comparison_df = competence_group_comparison(profile_df)
    validation_df = validate_competence_outputs(
        condition_df,
        result_df,
        metric_df,
        profile_df,
        definition_df,
        expected_evolved_min=1,
    )

    artifacts.extend(dataframe_to_artifacts(result_df, step_dir / "competence_proxy_results", results_dir / "e04_s11_competence_proxy_results"))
    artifacts.extend(dataframe_to_artifacts(tick_df, step_dir / "competence_proxy_tick_records", results_dir / "e04_s11_competence_proxy_tick_records"))
    artifacts.extend(dataframe_to_artifacts(profile_df, step_dir / "competence_policy_profiles", results_dir / "e04_s11_competence_policy_profiles"))
    artifacts.extend(dataframe_to_artifacts(metric_df, step_dir / "competence_metrics", results_dir / "e04_competence_metrics"))
    artifacts.extend(dataframe_to_artifacts(comparison_df, step_dir / "competence_group_comparison", results_dir / "e04_s11_competence_group_comparison"))
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "competence_proxy_validation", results_dir / "e04_s11_competence_proxy_validation"))

    figure_paths = plot_competence_radar(profile_df, figures_dir)
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
                        "claimBoundary": PROXY_SCOPE_NOTE,
                        "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "competence_proxy_validation", results_dir / "e04_s11_competence_proxy_validation"))

    code_artifacts = copy_code_artifacts(step_dir)
    artifacts.extend(code_artifacts)

    overall_row = comparison_df[comparison_df["metricId"] == "overallCompetenceProxy"]
    overall_delta = None if overall_row.empty else overall_row["enhancedMinusBaseline"].iloc[0]
    if overall_delta is None or pd.isna(overall_delta):
        outcome = "null"
    elif float(overall_delta) > 0.02:
        outcome = "supportive"
    elif float(overall_delta) < -0.02:
        outcome = "constraining/contradictory"
    else:
        outcome = "null"
    caveats = [
        "All S11 metrics are direct computational proxies and are not direct biological intelligence, cognition, agency, or sentience measurements.",
        "The held-out panel is intentionally small and stress-test oriented; it should not be read as broad open-ended generalization.",
        "Route diversity is measured by value-state transition hashes among successful runs, so it is sensitive to scheduler seeds and simulator granularity.",
        "Enhanced policies include S08 candidates replayed from logged local-only configs where available; missing or non-replayable candidates remain catalog-only.",
    ]
    recommended_next_action = "Chief review of S11 proxy evidence; do not start S12 from this run."
    reports = write_reports(
        step_dir=step_dir,
        result_df=result_df,
        profile_df=profile_df,
        comparison_df=comparison_df,
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
            "validationResultsPath": str(step_dir / "competence_proxy_validation.csv"),
        },
        "caveatsOrBlockers": caveats if success else caveats + ["At least one S11 validation check failed."],
        "recommendedNextAction": recommended_next_action,
        "experimentId": EXPERIMENT_ID,
        "title": STEP_TITLE,
        "outcomeClassification": outcome,
        "startedAt": config_payload["createdAt"],
        "completedAt": utc_now(),
        "runtimeSeconds": time.perf_counter() - started,
        "git": get_git_metadata(),
        "platform": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor()},
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(result_df)),
        "metricCount": int(len(metric_df)),
        "policyProfileCount": int(len(profile_df)),
        "policyGroupCounts": {str(key): int(value) for key, value in condition_df["policyGroup"].value_counts().to_dict().items()},
        "overallEnhancedMinusBaselineProxy": None if overall_delta is None or pd.isna(overall_delta) else float(overall_delta),
        "serialWorkerCount": 1,
        "proxyScopeNote": PROXY_SCOPE_NOTE,
        "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    write_json(status_path, status)
    manifest = artifact_manifest(artifacts_dir, [path for path in artifacts if path != manifest_path])
    write_json(manifest_path, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
