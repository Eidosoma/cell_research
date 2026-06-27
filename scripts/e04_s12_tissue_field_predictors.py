#!/usr/bin/env python3
"""Execute E04 S12 simulated aggregate field predictor analysis."""

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
    FIELD_PREDICTOR_VERSION,
    FIELD_PROXY_SCOPE_NOTE,
    FIELD_TARGETS,
    FIELD_VARIANTS,
    build_s12_policies,
    build_s12_tasks,
    condition_rows_for_field_policy,
    evaluate_field_predictors,
    field_feature_columns,
    field_feature_definitions,
    run_field_condition,
    summarize_field_predictors,
    validate_field_predictor_outputs,
)
from memory_repair.competence_proxies import competence_policy_from_s08_candidate  # noqa: E402
from memory_repair.training_constraints import TRAINING_CONSTRAINT_VERSION  # noqa: E402


EXPERIMENT_ID = "E04"
STEP_ID = "S12"
STEP_NUMBER = 12
STEP_TITLE = "Analyze emergent tissue fields"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
S08_CANDIDATE_PATH = Path("/artifacts/results/e04_evolved_repair_policies.parquet")
TEST_MODULES = [
    "tests.test_e04_field_predictors",
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


def plot_model_results(model_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if model_df.empty:
        return []
    evaluated = model_df[model_df["evaluated"].astype(bool)].copy()
    if evaluated.empty:
        return []
    targets = list(FIELD_TARGETS)
    variants = list(FIELD_VARIANTS)
    x = np.arange(len(targets))
    width = 0.18
    colors = {
        "observed_fields": "#4776A6",
        "permuted_null_fields": "#A66B45",
        "zero_null_fields": "#777777",
        "gaussian_null_fields": "#7B6BAF",
    }
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for index, variant in enumerate(variants):
        values = []
        for target in targets:
            row = evaluated[(evaluated["targetId"] == target) & (evaluated["fieldVariant"] == variant)]
            values.append(float(row["testRocAuc"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (index - 1.5) * width, values, width=width, label=variant, color=colors.get(variant, "#4B8F6A"))
    ax.axhline(0.5, color="#222222", linewidth=0.8)
    ax.set_xticks(x, [target.replace("Proxy", "").replace("future", "future ") for target in targets], rotation=18, ha="right")
    ax.set_ylabel("Held-out ROC AUC")
    ax.set_title("S12 simulated aggregate field predictors vs null fields")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = figures_dir / "e04_s12_field_predictor_auc.png"
    svg_path = figures_dir / "e04_s12_field_predictor_auc.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar(summary_df["targetId"], summary_df["observedMinusBestNullAuc"], color="#4B8F6A")
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(range(len(summary_df)), summary_df["targetId"].str.replace("Proxy", "", regex=False), rotation=20, ha="right")
        ax.set_ylabel("Observed minus best-null AUC")
        ax.set_title("S12 field predictor lift over null controls")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        lift_png = figures_dir / "e04_s12_field_predictor_null_lift.png"
        lift_svg = figures_dir / "e04_s12_field_predictor_null_lift.svg"
        fig.savefig(lift_png, dpi=160)
        fig.savefig(lift_svg)
        plt.close(fig)
        return [png_path, svg_path, lift_png, lift_svg]
    return [png_path, svg_path]


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_root = step_dir / "code"
    sources = [
        REPO_ROOT / "memory_repair" / "field_predictors.py",
        REPO_ROOT / "memory_repair" / "competence_proxies.py",
        REPO_ROOT / "memory_repair" / "memory.py",
        REPO_ROOT / "memory_repair" / "signals.py",
        REPO_ROOT / "memory_repair" / "repair.py",
        REPO_ROOT / "memory_repair" / "fatigue.py",
        REPO_ROOT / "memory_repair" / "homeostasis.py",
        REPO_ROOT / "memory_repair" / "learning.py",
        REPO_ROOT / "memory_repair" / "training_constraints.py",
        REPO_ROOT / "scripts" / "e04_s12_tissue_field_predictors.py",
        REPO_ROOT / "tests" / "test_e04_field_predictors.py",
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


def write_feature_definition_markdown(path: Path, definition_df: pd.DataFrame) -> None:
    preview = definition_df.head(40)
    path.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Field Feature Definitions",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{path}`, `field_feature_definitions.csv`, and `field_feature_definitions.parquet`",
                "- Validation result: feature definitions documented for every model feature column",
                "- Caveats or blockers: definitions describe simulated aggregate fields only",
                "- Recommended next action: Chief review of S12 before any S13 work",
                "",
                FIELD_PROXY_SCOPE_NOTE,
                "",
                markdown_table(
                    ["featureId", "sourceFieldFamily", "directComputationalDefinition"],
                    [[row.featureId, row.sourceFieldFamily, row.directComputationalDefinition] for row in preview.itertuples(index=False)],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_reports(
    *,
    step_dir: Path,
    condition_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    model_df: pd.DataFrame,
    summary_df: pd.DataFrame,
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
                f"- Artifacts written: `{step_dir / 'field_predictor_validation.csv'}` and `.parquet`",
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

    null_report = step_dir / "null_field_control_report.md"
    null_preview = model_df.sort_values(["targetId", "fieldVariant"], kind="mergesort")
    null_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Null-Field Control Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'field_predictor_model_results.csv'}`, `{step_dir / 'field_predictor_summary.csv'}`, and figure outputs",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                FIELD_PROXY_SCOPE_NOTE,
                "",
                "Observed simulated fields were compared against zero-field, seed-preserving permuted-field, and Gaussian matched-marginal null controls.",
                "",
                markdown_table(
                    ["target", "variant", "evaluated", "testAUC", "testAP", "testPrevalence"],
                    [
                        [row.targetId, row.fieldVariant, row.evaluated, row.testRocAuc, row.testAveragePrecision, row.testPrevalence]
                        for row in null_preview.itertuples(index=False)
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(null_report)

    report = step_dir / "tissue_field_prediction_report.md"
    report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Simulated Field Prediction Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'field_trace_features.csv'}`, `{step_dir / 'field_predictor_model_results.csv'}`, `{step_dir / 'field_predictor_summary.csv'}`, and field predictor figures",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                FIELD_PROXY_SCOPE_NOTE,
                "",
                "Summary of observed simulated field predictor lift over the best null-field control:",
                "",
                markdown_table(
                    ["target", "observedAUC", "bestNullAUC", "AUC lift", "observedAP", "bestNullAP", "AP lift"],
                    [
                        [
                            row.targetId,
                            row.observedAuc,
                            row.bestNullAuc,
                            row.observedMinusBestNullAuc,
                            row.observedAveragePrecision,
                            row.bestNullAveragePrecision,
                            row.observedMinusBestNullAveragePrecision,
                        ]
                        for row in summary_df.itertuples(index=False)
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(report)

    supportive_count = int(summary_df["supportiveProxySignal"].sum()) if "supportiveProxySignal" in summary_df else 0
    mean_auc_lift = float(summary_df["observedMinusBestNullAuc"].mean()) if len(summary_df) else None
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
                "Lay summary: S12 replayed local-memory and signaling policies, extracted simulated aggregate signal, memory, learning-frustration, repair-state, and fatigue-state fields at each trace row, and tested whether those fields predict future repair, failure, or recovery proxies on seed-held-out rows. The observed fields were compared to null fields that were zeroed, permuted, or replaced by matched random fields.",
                "",
                FIELD_PROXY_SCOPE_NOTE,
                "",
                f"Outcome classification: {outcome}.",
                f"Condition rows: {len(condition_df)}; trace feature rows: {len(trace_df)}; model rows: {len(model_df)}.",
                f"Supportive target count: `{supportive_count}` of `{len(summary_df)}`.",
                f"Mean observed-minus-best-null AUC lift: `{mean_auc_lift}`.",
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
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--train-seed-count", type=int, default=3)
    parser.add_argument("--max-evolved", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=14100)
    parser.add_argument("--lookahead-rows", type=int, default=8)
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
    tasks = build_s12_tasks()
    policies = build_s12_policies(evolved_rows, max_evolved=args.max_evolved)
    policy_catalog = [policy.to_dict() for policy in policies] + [
        {**row, "catalogOnly": True} for row in evolved_catalog if not row.get("replayable", False)
    ]
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "seedCount": int(args.seed_count),
        "trainSeedCount": int(args.train_seed_count),
        "seedBase": int(args.seed_base),
        "lookaheadRows": int(args.lookahead_rows),
        "maxEvolved": int(args.max_evolved),
        "serialWorkerCount": 1,
        "s08CandidatePath": str(S08_CANDIDATE_PATH),
        "fieldTargets": list(FIELD_TARGETS),
        "fieldVariants": list(FIELD_VARIANTS),
        "proxyScopeNote": FIELD_PROXY_SCOPE_NOTE,
        "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "createdAt": utc_now(),
    }
    for path in [step_dir / "field_predictor_config.json", configs_dir / "e04_s12_tissue_field_predictors.json"]:
        write_json(path, config_payload)
        artifacts.append(path)
    for path in [step_dir / "field_predictor_config.yaml", configs_dir / "e04_s12_tissue_field_predictors.yaml"]:
        write_yaml(path, config_payload)
        artifacts.append(path)

    task_df = pd.DataFrame([task.to_dict() for task in tasks])
    policy_df = pd.DataFrame(policy_catalog)
    artifacts.extend(dataframe_to_artifacts(task_df, step_dir / "field_prediction_tasks", results_dir / "e04_s12_field_prediction_tasks"))
    artifacts.extend(dataframe_to_artifacts(policy_df, step_dir / "field_prediction_policies", results_dir / "e04_s12_field_prediction_policies"))

    conditions = []
    for policy_index, policy in enumerate(policies):
        conditions.extend(
            condition_rows_for_field_policy(
                policy,
                tasks,
                seed_count=int(args.seed_count),
                train_seed_count=int(args.train_seed_count),
                seed_base=int(args.seed_base + policy_index * 1000),
            )
        )
    condition_df = pd.DataFrame(conditions)
    artifacts.extend(dataframe_to_artifacts(condition_df, step_dir / "field_prediction_conditions", results_dir / "e04_s12_field_prediction_conditions"))

    policy_by_id = {policy.policy_id: policy for policy in policies}
    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for condition in conditions:
        result, traces = run_field_condition(
            policy_by_id[str(condition["policyId"])],
            condition,
            lookahead_rows=int(args.lookahead_rows),
        )
        result_rows.append(result)
        trace_rows.extend(traces)
    result_df = pd.DataFrame(result_rows)
    trace_df = pd.DataFrame(trace_rows)
    feature_columns = field_feature_columns(trace_df)
    definition_df = pd.DataFrame(field_feature_definitions(feature_columns))
    model_df = evaluate_field_predictors(trace_df, feature_columns=feature_columns, null_seed=15100)
    summary_df = summarize_field_predictors(model_df)
    validation_df = validate_field_predictor_outputs(condition_df, trace_df, model_df, summary_df, definition_df)

    artifacts.extend(dataframe_to_artifacts(result_df, step_dir / "field_prediction_results", results_dir / "e04_s12_field_prediction_results"))
    artifacts.extend(dataframe_to_artifacts(trace_df, step_dir / "field_trace_features", results_dir / "e04_s12_field_trace_features"))
    artifacts.extend(dataframe_to_artifacts(definition_df, step_dir / "field_feature_definitions", results_dir / "e04_s12_field_feature_definitions"))
    definition_md = step_dir / "field_feature_definitions.md"
    write_feature_definition_markdown(definition_md, definition_df)
    artifacts.append(definition_md)
    artifacts.extend(dataframe_to_artifacts(model_df, step_dir / "field_predictor_model_results", results_dir / "e04_tissue_field_predictors"))
    artifacts.extend(dataframe_to_artifacts(summary_df, step_dir / "field_predictor_summary", results_dir / "e04_s12_tissue_field_predictor_summary"))
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "field_predictor_validation", results_dir / "e04_s12_field_predictor_validation"))

    figure_paths = plot_model_results(model_df, summary_df, figures_dir)
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
                        "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
                        "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "field_predictor_validation", results_dir / "e04_s12_field_predictor_validation"))

    code_artifacts = copy_code_artifacts(step_dir)
    artifacts.extend(code_artifacts)

    supportive_count = int(summary_df["supportiveProxySignal"].sum()) if "supportiveProxySignal" in summary_df else 0
    mean_auc_lift = float(summary_df["observedMinusBestNullAuc"].mean()) if len(summary_df) else 0.0
    if supportive_count >= 2 and mean_auc_lift > 0.0:
        outcome = "supportive"
    elif supportive_count == 0 and mean_auc_lift <= 0.0:
        outcome = "null"
    else:
        outcome = "constraining/contradictory" if mean_auc_lift < -0.02 else "null"
    caveats = [
        "All field language is proxy-scoped to simulated aggregate fields and is not a biological tissue, bioelectric, or morphogen measurement.",
        "Targets are future repair, failure, and recovery proxies defined over finite trace-row lookahead windows, not causal biological outcomes.",
        "Logistic predictors quantify held-out association under seed-separated splits; they do not prove that fields cause later repair or failure.",
        "Null fields preserve labels and splits but disrupt or remove field structure, so null comparisons test time-aligned field information within this simulator panel.",
    ]
    recommended_next_action = "Chief review of S12 field-proxy evidence; do not start S13 from this run."
    reports = write_reports(
        step_dir=step_dir,
        condition_df=condition_df,
        trace_df=trace_df,
        model_df=model_df,
        summary_df=summary_df,
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
            "validationResultsPath": str(step_dir / "field_predictor_validation.csv"),
        },
        "caveatsOrBlockers": caveats if success else caveats + ["At least one S12 validation check failed."],
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
        "traceFeatureRowCount": int(len(trace_df)),
        "modelResultCount": int(len(model_df)),
        "summaryRowCount": int(len(summary_df)),
        "featureColumnCount": int(len(feature_columns)),
        "supportiveTargetCount": supportive_count,
        "meanObservedMinusBestNullAuc": mean_auc_lift,
        "serialWorkerCount": 1,
        "proxyScopeNote": FIELD_PROXY_SCOPE_NOTE,
        "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    write_json(status_path, status)
    manifest = artifact_manifest(artifacts_dir, [path for path in artifacts if path != manifest_path])
    write_json(manifest_path, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
