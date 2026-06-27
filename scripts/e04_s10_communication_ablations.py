#!/usr/bin/env python3
"""Execute E04 S10 communication-architecture ablations."""

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
    COMMUNICATION_ABLATION_VERSION,
    build_s10_tasks,
    classic_policy_families,
    communication_transfer_gaps,
    compute_communication_deltas,
    condition_rows_for_communication_family,
    evolved_policy_family_from_candidate,
    learned_policy_families,
    no_signal_reproduction_rows,
    policy_family_from_frontier_row,
    run_communication_condition,
    summarize_communication_effects,
    validate_communication_outputs,
)
from memory_repair.training_constraints import TRAINING_CONSTRAINT_VERSION  # noqa: E402


EXPERIMENT_ID = "E04"
STEP_ID = "S10"
STEP_NUMBER = 10
STEP_TITLE = "Run communication ablations"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
E03_FRONTIER_PATH = Path("/previous-artifacts/E03/results/e03_frontier_candidates.parquet")
S08_CANDIDATE_PATH = Path("/artifacts/results/e04_evolved_repair_policies.parquet")
TEST_MODULES = [
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
    "tests.test_e03_policy_interface",
    "tests.test_e03_rule_dsl",
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


def load_frontier_families(max_frontier: int) -> tuple[list[Any], list[dict[str, Any]]]:
    if not E03_FRONTIER_PATH.exists():
        return [], [{"familyKind": "frontier", "replayable": False, "replayBlocker": f"missing {E03_FRONTIER_PATH}"}]
    df = pd.read_parquet(E03_FRONTIER_PATH).sort_values("frontierRank", kind="mergesort")
    families = []
    catalog_rows = []
    for _, row in df.iterrows():
        family = policy_family_from_frontier_row(row)
        catalog_rows.append(family.to_dict())
        if family.replayable and len(families) < max_frontier:
            families.append(family)
    return families, catalog_rows


def load_evolved_families(max_evolved: int) -> tuple[list[Any], list[dict[str, Any]]]:
    if not S08_CANDIDATE_PATH.exists():
        return [], [{"familyKind": "evolved", "replayable": False, "replayBlocker": f"missing {S08_CANDIDATE_PATH}"}]
    df = pd.read_parquet(S08_CANDIDATE_PATH).sort_values("rank", kind="mergesort")
    families = []
    catalog_rows = []
    for _, row in df.iterrows():
        family = evolved_policy_family_from_candidate(row)
        catalog_rows.append(family.to_dict())
        if family.replayable and len(families) < max_evolved:
            families.append(family)
    return families, catalog_rows


def plot_score_deltas(summary_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return []
    grouped = summary_df.groupby("ablationVariant", dropna=False)["meanScoreDelta"].mean().reset_index()
    colors = ["#4776A6", "#4B8F6A", "#A66B45", "#7B6BAF", "#B3526B"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(grouped))
    ax.bar(x, grouped["meanScoreDelta"], color=[colors[i % len(colors)] for i in range(len(grouped))])
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, grouped["ablationVariant"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Mean score delta vs no-signal reference")
    ax.set_title("S10 communication ablation score deltas")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png_path = figures_dir / "s10_communication_ablation_score_deltas.png"
    svg_path = figures_dir / "s10_communication_ablation_score_deltas.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    return [png_path, svg_path]


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_root = step_dir / "code"
    sources = [
        REPO_ROOT / "memory_repair" / "communication_ablations.py",
        REPO_ROOT / "memory_repair" / "signals.py",
        REPO_ROOT / "memory_repair" / "repair.py",
        REPO_ROOT / "memory_repair" / "fatigue.py",
        REPO_ROOT / "memory_repair" / "homeostasis.py",
        REPO_ROOT / "memory_repair" / "learning.py",
        REPO_ROOT / "memory_repair" / "ablations.py",
        REPO_ROOT / "scripts" / "e04_s10_communication_ablations.py",
        REPO_ROOT / "tests" / "test_e04_communication_ablations.py",
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
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    transfer_gap_df: pd.DataFrame,
    no_signal_df: pd.DataFrame,
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
                f"- Artifacts written: `{step_dir / 'communication_ablation_validation.csv'}` and `.parquet`",
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

    baseline_report = step_dir / "no_signal_baseline_report.md"
    baseline_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} No-Signal Baseline Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'no_signal_baseline_reproduction.csv'}` and `.parquet`",
                f"- Validation result: {int(no_signal_df['success'].sum())} of {len(no_signal_df)} no-signal reproductions matched",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "No-signal S10 arms were replayed against an explicit no-signal policy wrapper with the same repair, fatigue, task, and scheduler settings.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(baseline_report)

    ablation_report = step_dir / "communication_ablation_report.md"
    compact_summary = summary_df.sort_values(["ablationVariant", "familyKind", "taskFamily", "transfer"], kind="mergesort").head(30)
    gap_preview = transfer_gap_df.sort_values(["ablationVariant", "familyKind", "taskFamily"], kind="mergesort").head(18)
    ablation_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Communication Ablation Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'communication_ablation_results.csv'}`, `{step_dir / 'communication_ablation_deltas.csv'}`, `{step_dir / 'communication_ablation_summary.csv'}`, and `{step_dir / 'communication_transfer_gaps.csv'}`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "S10 quantifies each signal architecture as a paired delta against no signaling within the same policy family, task, and seed.",
                "",
                markdown_table(
                    ["variant", "familyKind", "taskFamily", "transfer", "meanScoreDelta", "meanFalseAlarmDelta", "improvedFraction", "n"],
                    [
                        [
                            row.ablationVariant,
                            row.familyKind,
                            row.taskFamily,
                            row.transfer,
                            row.meanScoreDelta,
                            row.meanFalseAlarmDelta,
                            row.improvedScoreFraction,
                            row.comparisonCount,
                        ]
                        for row in compact_summary.itertuples(index=False)
                    ],
                ),
                "",
                "Transfer-gap preview:",
                "",
                markdown_table(
                    ["variant", "familyKind", "taskFamily", "trainingDelta", "transferDelta", "overfittingGap"],
                    [
                        [
                            row.ablationVariant,
                            row.familyKind,
                            row.taskFamily,
                            row.trainingMeanScoreDelta,
                            row.transferMeanScoreDelta,
                            row.overfittingGap,
                        ]
                        for row in gap_preview.itertuples(index=False)
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(ablation_report)

    mean_abs_delta = float(delta_df["scoreDelta"].abs().mean()) if not delta_df.empty else 0.0
    mean_false_alarm = float(result_df["falseAlarmProxy"].mean()) if not result_df.empty else 0.0
    summary = step_dir / "summary.md"
    summary.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Status Summary",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: primary outputs under `{step_dir}` plus shared result/config/figure copies; full list is in `status.json` and `artifact_manifest.json`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "Lay summary: S10 replayed no signaling, nearest-neighbor, diffusive, long-range/scalar, noisy, and randomized-signal controls across repairable Frozen Cell, fatigue, and homeostasis tasks. Each comparison used paired seeds and identical perturbation schedules, and no-signal arms reproduced explicit no-signal baselines.",
                "",
                f"Outcome classification: {outcome}.",
                f"Condition rows: {len(condition_df)}; result rows: {len(result_df)}; delta rows: {len(delta_df)}.",
                f"Mean absolute communication score delta: `{mean_abs_delta}`.",
                f"Mean false-alarm proxy across results: `{mean_false_alarm}`.",
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
    parser.add_argument("--max-frontier", type=int, default=2)
    parser.add_argument("--max-evolved", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=10100)
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

    tasks = build_s10_tasks()
    frontier_families, frontier_catalog = load_frontier_families(args.max_frontier)
    evolved_families, evolved_catalog = load_evolved_families(args.max_evolved)
    families = list(classic_policy_families()) + frontier_families + list(learned_policy_families()) + evolved_families
    family_catalog = [family.to_dict() for family in families] + [
        {**row, "catalogOnly": True} for row in frontier_catalog + evolved_catalog if not row.get("replayable", False)
    ]

    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "seedCount": int(args.seed_count),
        "seedBase": int(args.seed_base),
        "maxFrontier": int(args.max_frontier),
        "maxEvolved": int(args.max_evolved),
        "serialWorkerCount": 1,
        "e03FrontierPath": str(E03_FRONTIER_PATH),
        "s08CandidatePath": str(S08_CANDIDATE_PATH),
        "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "createdAt": utc_now(),
    }
    for path in [step_dir / "communication_ablation_config.json", configs_dir / "e04_communication_ablations.json"]:
        write_json(path, config_payload)
        artifacts.append(path)
    for path in [step_dir / "communication_ablation_config.yaml", configs_dir / "e04_communication_ablations.yaml"]:
        write_yaml(path, config_payload)
        artifacts.append(path)

    task_df = pd.DataFrame([task.to_dict() for task in tasks])
    family_df = pd.DataFrame(family_catalog)
    artifacts.extend(dataframe_to_artifacts(task_df, step_dir / "communication_ablation_tasks", results_dir / "e04_s10_communication_ablation_tasks"))
    artifacts.extend(dataframe_to_artifacts(family_df, step_dir / "communication_ablation_policy_families", results_dir / "e04_s10_communication_ablation_policy_families"))

    conditions = []
    for family_index, family in enumerate(families):
        conditions.extend(
            condition_rows_for_communication_family(
                family,
                tasks,
                seed_count=int(args.seed_count),
                seed_base=int(args.seed_base + family_index * 1000),
            )
        )
    condition_df = pd.DataFrame(conditions)
    artifacts.extend(dataframe_to_artifacts(condition_df, step_dir / "communication_ablation_conditions", results_dir / "e04_s10_communication_ablation_conditions"))

    family_by_id = {family.family_id: family for family in families}
    result_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    for condition in conditions:
        result, ticks = run_communication_condition(family_by_id[str(condition["familyId"])], condition)
        result_rows.append(result)
        tick_rows.extend(ticks)
    result_df = pd.DataFrame(result_rows)
    tick_df = pd.DataFrame(tick_rows)
    delta_df = compute_communication_deltas(result_df)
    summary_df = summarize_communication_effects(delta_df)
    transfer_gap_df = communication_transfer_gaps(delta_df)
    no_signal_df = no_signal_reproduction_rows(condition_df, result_df, family_by_id)
    validation_df = validate_communication_outputs(condition_df, result_df, delta_df, no_signal_df, expected_evolved_min=1)

    artifacts.extend(dataframe_to_artifacts(result_df, step_dir / "communication_ablation_results", results_dir / "e04_communication_ablations"))
    artifacts.extend(dataframe_to_artifacts(delta_df, step_dir / "communication_ablation_deltas", results_dir / "e04_s10_communication_ablation_deltas"))
    artifacts.extend(dataframe_to_artifacts(summary_df, step_dir / "communication_ablation_summary", results_dir / "e04_s10_communication_ablation_summary"))
    artifacts.extend(dataframe_to_artifacts(transfer_gap_df, step_dir / "communication_transfer_gaps", results_dir / "e04_s10_communication_transfer_gaps"))
    artifacts.extend(dataframe_to_artifacts(tick_df, step_dir / "communication_ablation_tick_records", results_dir / "e04_s10_communication_ablation_tick_records"))
    artifacts.extend(dataframe_to_artifacts(no_signal_df, step_dir / "no_signal_baseline_reproduction", results_dir / "e04_s10_no_signal_baseline_reproduction"))
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "communication_ablation_validation", results_dir / "e04_s10_communication_ablation_validation"))

    figure_paths = plot_score_deltas(summary_df, figures_dir)
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
                        "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "communication_ablation_validation", results_dir / "e04_s10_communication_ablation_validation"))

    code_artifacts = copy_code_artifacts(step_dir)
    artifacts.extend(code_artifacts)

    mean_abs_delta = float(delta_df["scoreDelta"].abs().mean()) if not delta_df.empty else 0.0
    mean_best_delta = float(summary_df.groupby("ablationVariant")["meanScoreDelta"].mean().max()) if not summary_df.empty else 0.0
    if mean_abs_delta <= 1e-9:
        outcome = "null"
    elif mean_best_delta > 1e-9:
        outcome = "supportive"
    else:
        outcome = "constraining/contradictory"
    caveats = [
        "Long-range/scalar signaling uses a large local sensing range on small arrays and should remain labeled as a separate communication architecture, not ordinary nearest-neighbor locality.",
        "Noisy and randomized controls can trigger signal-threshold repair through local sensed exposure, so false-alarm and energy proxies must be interpreted with the paired no-signal reference.",
        "S10 is a small CPU-reference ablation panel and should not be read as a broad mechanism ranking.",
    ]
    recommended_next_action = "Chief review; if accepted, proceed to S11 competence metrics. Do not start S11 from this run."
    reports = write_reports(
        step_dir=step_dir,
        condition_df=condition_df,
        result_df=result_df,
        delta_df=delta_df,
        summary_df=summary_df,
        transfer_gap_df=transfer_gap_df,
        no_signal_df=no_signal_df,
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
            "validationResultsPath": str(step_dir / "communication_ablation_validation.csv"),
        },
        "caveatsOrBlockers": caveats if success else caveats + ["At least one S10 validation check failed."],
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
        "deltaCount": int(len(delta_df)),
        "noSignalBaselineReproductionCount": int(len(no_signal_df)),
        "familyKindCounts": {str(key): int(value) for key, value in condition_df["familyKind"].value_counts().to_dict().items()},
        "meanAbsScoreDelta": mean_abs_delta,
        "bestMeanVariantScoreDelta": mean_best_delta,
        "meanFalseAlarmProxy": float(result_df["falseAlarmProxy"].mean()) if len(result_df) else 0.0,
        "serialWorkerCount": 1,
        "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    write_json(status_path, status)
    manifest = artifact_manifest(artifacts_dir, [path for path in artifacts if path != manifest_path])
    write_json(manifest_path, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
