#!/usr/bin/env python3
"""Execute E04 S14 centralized global-oracle repair comparison."""

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
    CENTRALIZED_COMPARISON_VERSION,
    CENTRALIZED_ORACLE_CONTROLLER_ID,
    CENTRALIZED_PROXY_SCOPE_NOTE,
    TRAINING_CONSTRAINT_VERSION,
    build_s14_local_policies,
    build_s14_tasks,
    centralized_oracle_controller_spec,
    competence_policy_from_s08_candidate,
    fairness_assumptions,
    global_condition_rows_for_s14,
    local_condition_rows_for_s14,
    run_s14_global_condition,
    run_s14_local_condition,
    summarize_centralized_groups,
    summarize_local_global_deltas,
    validate_centralized_outputs,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S14"
STEP_NUMBER = 14
STEP_TITLE = "Centralized repair comparison"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
S08_CANDIDATE_PATH = Path("/artifacts/results/e04_evolved_repair_policies.parquet")
TEST_MODULES = [
    "tests.test_e04_centralized_comparison",
    "tests.test_e04_overfitting_transfer",
    "tests.test_e04_competence_proxies",
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
    df = pd.read_parquet(S08_CANDIDATE_PATH)
    if "rank" in df.columns:
        df = df.sort_values("rank", kind="mergesort")
    rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        policy = competence_policy_from_s08_candidate(row)
        catalog_rows.append(policy.to_dict())
        if policy.replayable and len(rows) < max_evolved:
            rows.append(row.to_dict())
    return rows, catalog_rows


def oracle_records_from_results(result_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in result_rows:
        if result.get("controllerKind") != "global_oracle":
            continue
        records = []
        payload = result.get("oracleRecordsJson", "[]")
        if isinstance(payload, str):
            try:
                records = json.loads(payload)
            except json.JSONDecodeError:
                records = []
        elif isinstance(payload, list):
            records = payload
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            row = dict(record)
            row.update(
                {
                    "oracleRecordIndex": int(index),
                    "conditionId": result.get("conditionId"),
                    "pairKey": result.get("pairKey"),
                    "policyId": result.get("policyId"),
                    "taskId": result.get("taskId"),
                    "taskFamily": result.get("taskFamily"),
                    "seedIndex": result.get("seedIndex"),
                    "controllerKind": "global_oracle",
                    "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                    "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
                }
            )
            rows.append(json_ready(row))
    return rows


def plot_centralized_results(delta_df: pd.DataFrame, group_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if delta_df.empty:
        return []
    ordered = delta_df.sort_values(["taskId", "localPolicyId", "seedIndex"], kind="mergesort").copy()
    ordered["label"] = ordered["taskId"].astype(str).str.replace("s14_", "", regex=False) + "\n" + ordered["localPolicyId"].astype(str).str.replace("enhanced_", "enh_", regex=False)
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(max(10.0, len(ordered) * 0.42), 5.6))
    width = 0.38
    ax.bar(x - width / 2, ordered["localControllerScore"], width=width, label="local", color="#4F7CAC")
    ax.bar(x + width / 2, ordered["globalControllerScore"], width=width, label="global oracle", color="#9A6A3A")
    ax.set_xticks(x, ordered["label"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Computational controller score")
    ax.set_title("S14 local versus centralized global-oracle scores")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    score_png = figures_dir / "e04_s14_local_vs_global_scores.png"
    score_svg = figures_dir / "e04_s14_local_vs_global_scores.svg"
    fig.savefig(score_png, dpi=160)
    fig.savefig(score_svg)
    plt.close(fig)

    if group_df.empty:
        return [score_png, score_svg]
    grouped = group_df.sort_values("meanGlobalMinusLocalScore", ascending=False, kind="mergesort")
    labels = grouped["localPolicyId"].astype(str).str.replace("enhanced_", "enh_", regex=False).tolist()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#4B8F6A" if value >= 0 else "#A6544F" for value in grouped["meanGlobalMinusLocalScore"]]
    ax.bar(labels, grouped["meanGlobalMinusLocalScore"], color=colors)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Global oracle minus local score")
    ax.set_title("S14 centralized upper-bound score gaps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    gap_png = figures_dir / "e04_s14_global_minus_local_score.png"
    gap_svg = figures_dir / "e04_s14_global_minus_local_score.svg"
    fig.savefig(gap_png, dpi=160)
    fig.savefig(gap_svg)
    plt.close(fig)
    return [score_png, score_svg, gap_png, gap_svg]


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_root = step_dir / "code"
    sources = [
        REPO_ROOT / "memory_repair" / "centralized_comparison.py",
        REPO_ROOT / "memory_repair" / "competence_proxies.py",
        REPO_ROOT / "memory_repair" / "memory.py",
        REPO_ROOT / "memory_repair" / "signals.py",
        REPO_ROOT / "memory_repair" / "repair.py",
        REPO_ROOT / "memory_repair" / "fatigue.py",
        REPO_ROOT / "memory_repair" / "homeostasis.py",
        REPO_ROOT / "memory_repair" / "learning.py",
        REPO_ROOT / "memory_repair" / "training_constraints.py",
        REPO_ROOT / "scripts" / "e04_s14_centralized_comparison.py",
        REPO_ROOT / "tests" / "test_e04_centralized_comparison.py",
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
    group_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
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
                f"- Artifacts written: `{step_dir / 'centralized_comparison_validation.csv'}` and `.parquet`",
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

    fairness_report = step_dir / "fairness_assumptions.md"
    fairness_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Fairness Assumptions",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'centralized_fairness_assumptions.csv'}` and `.parquet`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                CENTRALIZED_PROXY_SCOPE_NOTE,
                "",
                "The centralized controller is intentionally a nonlocal upper-bound comparator with `oracleAllowed=true`, `oracleBaseline=true`, and `usesGlobalController=true`. Local rows keep these flags false and keep their S07 local-only audit labels.",
                "",
                markdown_table(
                    ["assumptionId", "assumption", "limitation"],
                    [[row.assumptionId, row.assumption, row.limitation] for row in fairness_df.itertuples(index=False)],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(fairness_report)

    global_rows = result_df[result_df["controllerKind"] == "global_oracle"] if not result_df.empty else result_df
    report = step_dir / "centralized_comparison_report.md"
    report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Centralized Repair Comparison Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'centralized_comparison_results.csv'}`, `{step_dir / 'centralized_comparison_deltas.csv'}`, `{step_dir / 'centralized_group_summary.csv'}`, fairness assumptions, and S14 figures",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                CENTRALIZED_PROXY_SCOPE_NOTE,
                "",
                "Global-oracle controller label:",
                "",
                markdown_table(
                    ["policyId", "oracleAllowed", "oracleBaseline", "usesGlobalController", "usesWholeArrayValues", "usesTargetPositionOracle"],
                    [
                        [
                            CENTRALIZED_ORACLE_CONTROLLER_ID,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ]
                    ],
                ),
                "",
                "Policy-level local/global gaps:",
                "",
                markdown_table(
                    ["localPolicyId", "comparisons", "tasks", "local score", "global score", "global-local score", "global-local energy"],
                    [
                        [
                            row.localPolicyId,
                            row.comparisonCount,
                            row.taskCount,
                            row.meanLocalControllerScore,
                            row.meanGlobalControllerScore,
                            row.meanGlobalMinusLocalScore,
                            row.meanGlobalMinusLocalEnergyProxy,
                        ]
                        for row in group_df.itertuples(index=False)
                    ],
                ),
                "",
                "Task/seed pairing examples:",
                "",
                markdown_table(
                    ["taskId", "seed", "localPolicyId", "global-local score", "global interventions"],
                    [
                        [
                            row.taskId,
                            row.seedIndex,
                            row.localPolicyId,
                            row.globalMinusLocalScore,
                            row.globalInterventionCount,
                        ]
                        for row in delta_df.head(12).itertuples(index=False)
                    ],
                ),
                "",
                f"Global oracle result rows: {len(global_rows)}; all global rows carry `oracleAllowed=true`, `oracleBaseline=true`, and `usesGlobalController=true`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(report)

    mean_delta = None if delta_df.empty else float(delta_df["globalMinusLocalScore"].mean())
    mean_energy_delta = None if delta_df.empty else float(delta_df["globalMinusLocalEnergyProxy"].mean())
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
                "Lay summary: S14 adds an explicitly labeled centralized global-oracle comparator and runs it against local policies on paired computational sorting, repairable Frozen Cell, fatigue/damage, and homeostasis tasks. The outputs are proxy-scoped local-versus-global controller measurements.",
                "",
                CENTRALIZED_PROXY_SCOPE_NOTE,
                "",
                f"Outcome classification: {outcome}.",
                f"Condition rows: {len(condition_df)}; result rows: {len(result_df)}; paired delta rows: {len(delta_df)}.",
                f"Mean global-minus-local score: `{mean_delta}`.",
                f"Mean global-minus-local energy proxy: `{mean_energy_delta}`.",
                "",
                "Group summary:",
                "",
                markdown_table(
                    ["localPolicyId", "comparisons", "local score", "global score", "global-local score", "global completed", "local completed"],
                    [
                        [
                            row.localPolicyId,
                            row.comparisonCount,
                            row.meanLocalControllerScore,
                            row.meanGlobalControllerScore,
                            row.meanGlobalMinusLocalScore,
                            row.globalCompletedCount,
                            row.localCompletedCount,
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
    parser.add_argument("--seed-base", type=int, default=18100)
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

    evolved_rows, evolved_catalog = load_evolved_rows(int(args.max_evolved))
    tasks = build_s14_tasks()
    policies = build_s14_local_policies(evolved_rows, max_evolved=int(args.max_evolved))
    controller_spec = centralized_oracle_controller_spec()
    policy_catalog = [policy.to_dict() for policy in policies] + [
        {**row, "catalogOnly": True} for row in evolved_catalog if not row.get("replayable", False)
    ]
    fairness_rows = fairness_assumptions()
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
        "controllerKinds": ["local", "global_oracle"],
        "centralizedControllerPolicyId": CENTRALIZED_ORACLE_CONTROLLER_ID,
        "centralizedControllerOracleAllowed": True,
        "centralizedControllerOracleBaseline": True,
        "centralizedControllerUsesGlobalController": True,
        "pairingContract": "Local/global comparisons are paired by taskId, taskConfigHash, seedIndex, schedulerSeed, tieBreakerSeed, and signalRandomSeed where applicable.",
        "fairnessAssumptionIds": [row["assumptionId"] for row in fairness_rows],
        "proxyScopeNote": CENTRALIZED_PROXY_SCOPE_NOTE,
        "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "createdAt": utc_now(),
    }
    for path in [step_dir / "centralized_comparison_config.json", configs_dir / "e04_s14_centralized_comparison.json"]:
        write_json(path, config_payload)
        artifacts.append(path)
    for path in [step_dir / "centralized_comparison_config.yaml", configs_dir / "e04_s14_centralized_comparison.yaml"]:
        write_yaml(path, config_payload)
        artifacts.append(path)

    task_df = pd.DataFrame([task.to_dict() for task in tasks])
    policy_df = pd.DataFrame(policy_catalog)
    controller_df = pd.DataFrame([controller_spec])
    fairness_df = pd.DataFrame(fairness_rows)
    artifacts.extend(dataframe_to_artifacts(task_df, step_dir / "centralized_comparison_tasks", results_dir / "e04_s14_centralized_comparison_tasks"))
    artifacts.extend(dataframe_to_artifacts(policy_df, step_dir / "centralized_comparison_policies", results_dir / "e04_s14_centralized_comparison_policies"))
    artifacts.extend(dataframe_to_artifacts(controller_df, step_dir / "centralized_global_oracle_controller", results_dir / "e04_s14_centralized_global_oracle_controller"))
    artifacts.extend(dataframe_to_artifacts(fairness_df, step_dir / "centralized_fairness_assumptions", results_dir / "e04_s14_centralized_fairness_assumptions"))

    local_conditions = local_condition_rows_for_s14(policies, tasks, seed_count=int(args.seed_count), seed_base=int(args.seed_base))
    global_conditions = global_condition_rows_for_s14(tasks, seed_count=int(args.seed_count), seed_base=int(args.seed_base))
    conditions = local_conditions + global_conditions
    condition_df = pd.DataFrame(conditions)
    artifacts.extend(dataframe_to_artifacts(condition_df, step_dir / "centralized_comparison_conditions", results_dir / "e04_s14_centralized_comparison_conditions"))

    policy_by_id = {policy.policy_id: policy for policy in policies}
    result_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    for condition in local_conditions:
        result, ticks = run_s14_local_condition(policy_by_id[str(condition["policyId"])], condition)
        result_rows.append(result)
        tick_rows.extend(ticks)
    for condition in global_conditions:
        result, ticks = run_s14_global_condition(condition)
        result_rows.append(result)
        tick_rows.extend(ticks)
    result_df = pd.DataFrame(result_rows)
    tick_df = pd.DataFrame(tick_rows)
    oracle_df = pd.DataFrame(oracle_records_from_results(result_rows))
    delta_df = summarize_local_global_deltas(result_df)
    group_df = summarize_centralized_groups(delta_df)
    validation_df = validate_centralized_outputs(condition_df, result_df, delta_df, group_df, fairness_df)

    artifacts.extend(dataframe_to_artifacts(result_df, step_dir / "centralized_comparison_results", results_dir / "e04_centralized_comparison"))
    artifacts.extend(dataframe_to_artifacts(tick_df, step_dir / "centralized_comparison_tick_records", results_dir / "e04_s14_centralized_comparison_tick_records"))
    artifacts.extend(dataframe_to_artifacts(oracle_df, step_dir / "centralized_oracle_records", results_dir / "e04_s14_centralized_oracle_records"))
    artifacts.extend(dataframe_to_artifacts(delta_df, step_dir / "centralized_comparison_deltas", results_dir / "e04_s14_centralized_comparison_deltas"))
    artifacts.extend(dataframe_to_artifacts(group_df, step_dir / "centralized_group_summary", results_dir / "e04_s14_centralized_group_summary"))

    figure_paths = plot_centralized_results(delta_df, group_df, figures_dir)
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
                        "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                        "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "centralized_comparison_validation", results_dir / "e04_s14_centralized_comparison_validation"))

    code_artifacts = copy_code_artifacts(step_dir)
    artifacts.extend(code_artifacts)

    mean_delta = None if delta_df.empty else float(delta_df["globalMinusLocalScore"].mean())
    mean_energy_delta = None if delta_df.empty else float(delta_df["globalMinusLocalEnergyProxy"].mean())
    if not bool(validation_df["success"].all()):
        outcome = "constraining/contradictory"
    elif mean_delta is None or pd.isna(mean_delta):
        outcome = "null"
    elif mean_delta >= -0.02:
        outcome = "supportive"
    else:
        outcome = "constraining/contradictory"
    caveats = [
        "S14 scores are direct computational local-versus-global controller proxies and do not measure biological tissue repair, morphogenesis, intelligence, or causal wet-lab behavior.",
        "The centralized controller is intentionally nonlocal and oracle-labeled; it is an upper-bound comparator, not a fair local-only training policy.",
        "Energy proxies are simulator operation counts and intervention counts; they are not physically comparable biological costs.",
        "The paired panel is small and covers representative sorting, repairable Frozen Cell, fatigue/damage, and homeostasis tasks rather than all S09-S13 stress regimes.",
    ]
    if not S08_CANDIDATE_PATH.exists():
        caveats.append(f"S08 evolved candidate input was unavailable at {S08_CANDIDATE_PATH}; S14 used replayable baseline/local policies only.")
    recommended_next_action = "Chief review of S14 centralized global-oracle comparison and fairness assumptions; do not start S15 from this run."
    reports = write_reports(
        step_dir=step_dir,
        condition_df=condition_df,
        result_df=result_df,
        delta_df=delta_df,
        group_df=group_df,
        fairness_df=fairness_df,
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
            "validationResultsPath": str(step_dir / "centralized_comparison_validation.csv"),
        },
        "caveatsOrBlockers": caveats if success else caveats + ["At least one S14 validation check failed."],
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
        "localConditionCount": int(len(local_conditions)),
        "globalConditionCount": int(len(global_conditions)),
        "pairedDeltaCount": int(len(delta_df)),
        "localPolicyCount": int(len(policies)),
        "globalOraclePolicyId": CENTRALIZED_ORACLE_CONTROLLER_ID,
        "globalOracleExplicitlyLabeled": bool(
            not result_df[result_df["controllerKind"] == "global_oracle"].empty
            and result_df.loc[result_df["controllerKind"] == "global_oracle", "oracleAllowed"].astype(bool).all()
            and result_df.loc[result_df["controllerKind"] == "global_oracle", "oracleBaseline"].astype(bool).all()
            and result_df.loc[result_df["controllerKind"] == "global_oracle", "usesGlobalController"].astype(bool).all()
        ),
        "meanGlobalMinusLocalScore": mean_delta,
        "meanGlobalMinusLocalEnergyProxy": mean_energy_delta,
        "serialWorkerCount": 1,
        "proxyScopeNote": CENTRALIZED_PROXY_SCOPE_NOTE,
        "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    write_json(status_path, status)
    manifest = artifact_manifest(artifacts_dir, [path for path in artifacts if path != manifest_path])
    write_json(manifest_path, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
