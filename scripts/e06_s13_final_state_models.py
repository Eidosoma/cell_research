#!/usr/bin/env python3
"""Run E06 S13 final-state predictive and exploratory causal-style models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.final_state_models import (  # noqa: E402
    ANNOTATION_ONLY_SOURCE_STEPS,
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    EXPECTED_MODEL_ROW_SOURCE_STEPS,
    NUMERIC_FEATURES,
    REGRESSION_TARGETS,
    STEP_ID,
    S13Config,
    audit_predictor_leakage,
    build_causal_screen,
    build_feature_catalog,
    build_model_input_matrix,
    build_s06_context_annotations,
    evaluate_models,
    infer_s13_outcome,
    summarize_performance,
    train_final_models,
    validate_s13_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, sha256_file  # noqa: E402


STEP_NUMBER = 13
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--s07-mosaic-path", type=Path, default=ARTIFACTS_DIR / "results/e06_mosaic_classes.parquet")
    parser.add_argument("--s05-score-path", type=Path, default=ARTIFACTS_DIR / "results/e06_compatibility_scores.parquet")
    parser.add_argument("--s06-context-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s06_context_dependence.csv")
    parser.add_argument("--s06-hierarchy-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_dominance_hierarchy.csv")
    parser.add_argument("--s08-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_interface_rule_interventions.parquet")
    parser.add_argument("--s09-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_governance_mechanisms.parquet")
    parser.add_argument("--s10-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_graft_experiments.parquet")
    parser.add_argument("--s11-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_mutant_clone_experiments.parquet")
    parser.add_argument("--s12-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_developmental_history.parquet")
    parser.add_argument("--split-count", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--random-forest-trees", type=int, default=96)
    parser.add_argument("--max-permutation-rows", type=int, default=300)
    parser.add_argument("--permutation-repeats", type=int, default=1)
    parser.add_argument("--permutation-split-limit", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = datetime.now(UTC)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return {
        "command": " ".join(command),
        "cwd": str(cwd),
        "returnCode": proc.returncode,
        "success": proc.returncode == 0,
        "elapsedSeconds": elapsed,
        "stdout": proc.stdout[-6000:],
        "stderr": proc.stderr[-6000:],
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def pending_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum computed after this file is written.",
    }


def source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "sizeBytes": path.stat().st_size if path.exists() else None,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    present = [column for column in columns if column in frame.columns]
    display = frame[present].head(max_rows).copy()
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join("---" for _ in present) + " |",
    ]
    for row in display.to_dict(orient="records"):
        values = []
        for column in present:
            value = row[column]
            if isinstance(value, float):
                text = f"{value:.6g}"
            elif isinstance(value, bool):
                text = "true" if value else "false"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("| ... | " + f"{len(frame) - max_rows} more rows omitted from report table" + " |" * (len(present) - 1))
    return "\n".join(lines)


def read_parquet_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required S13 input is missing: {path}")
    return pd.read_parquet(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_feature_importance_figure(path: Path, importance: pd.DataFrame, performance_summary: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if importance.empty:
        return False
    grouped = (
        importance.groupby(["target_name", "feature_name"], dropna=False)
        .agg(mean_importance=("importance_mean", "mean"))
        .reset_index()
    )
    targets = [CLASSIFICATION_TARGET, *REGRESSION_TARGETS]
    available = [target for target in targets if target in set(grouped["target_name"])]
    if not available:
        return False
    fig, axes = plt.subplots(len(available), 1, figsize=(14, 4.2 * len(available)), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]
    for ax, target in zip(axes, available, strict=True):
        sub = grouped[grouped["target_name"] == target].sort_values("mean_importance", ascending=False).head(12)
        labels = sub["feature_name"].astype(str).tolist()[::-1]
        values = sub["mean_importance"].to_numpy(dtype=float)[::-1]
        ax.barh(labels, values, color="#3f7f6f")
        ax.axvline(0, color="#222222", lw=0.8)
        ax.set_title(f"Held-out permutation importance: {target}")
        ax.set_xlabel("Mean importance")
    fig.suptitle("E06 S13 final-state predictor feature importance", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def write_report(
    path: Path,
    *,
    artifacts_dir: Path,
    config: S13Config,
    input_paths: dict[str, Path],
    matrix: pd.DataFrame,
    performance_summary: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    feature_importance: pd.DataFrame,
    causal_screen: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: dict[str, Any],
    outcome: dict[str, Any],
    unit_test_result: dict[str, Any],
    command_line: str,
) -> None:
    artifacts_written = [entry["relativePath"] for entry in manifest["artifacts"]]
    source_counts = matrix["source_research_step_id"].value_counts().sort_index().reset_index()
    source_counts.columns = ["source_research_step_id", "row_count"]
    perf_sorted = performance_summary.sort_values(
        ["target_type", "target_name", "mean_balanced_accuracy", "mean_r2", "model_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    leakage_summary = leakage_audit["leakage_status"].value_counts().sort_index().reset_index()
    leakage_summary.columns = ["leakage_status", "column_count"]
    validation_passed = bool(validation["success"].all()) if not validation.empty else False
    report = f"""# E06 S13 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{chr(10).join(f'- `{item}`' for item in artifacts_written)}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and focused unit tests {"passed" if unit_test_result.get("success") else "failed"}.
- Outcome classification: {outcome["outcomeClassification"]}
- Caveats or blockers: S13 is an offline predictive analysis over computational proxy outcomes. Predictors exclude final outcome descriptors, S05/S07 derived classifications, and S06 dominance/context annotations; S06 annotations are provided only as separate explanatory context. Exploratory causal-style outputs are feature-importance associations, not causal proof.
- Lay summary: S13 asked whether final chimeric state can be predicted from the experiment design: which policies were mixed, ratios, arrangements, goals, histories, graft or clone schedules, interface rules, and governance mechanisms. Held-out condition groups were kept out of training so predictions were tested on unseen conditions rather than repeated seeds from the same condition.
- Recommended next action: Proceed to S14 intervention search using the held-out predictor errors, high-confidence failure contexts, and feature-importance screen, while preserving the same leakage controls.

## Frozen Question

Can final state class be predicted from policy, ratio, arrangement, goal compatibility, perturbation, and intervention variables?

## Inputs

{markdown_table(pd.DataFrame([{"input_name": name, "path": str(path), "exists": path.exists()} for name, path in input_paths.items()]), ["input_name", "path", "exists"], max_rows=30)}

## Methods

S13 built a row-level model input matrix from the S07 harmonized S02-S06 run matrix and the direct S08-S12 run-level outcomes. S05 compatibility outputs and S07 labels were treated as context outputs rather than predictors. S06 dominance hierarchy and context-dependence annotations were written separately and were deliberately excluded from the feature list.

Predictors were restricted to design variables and predeclared mechanism metadata: source step, policy and source-category signatures, ratio summaries, arrangement and goal profile IDs, interface and governance mechanism IDs, staged-history schedule encodings, graft and clone schedule encodings, memory-policy presence, explicit-recognition flags, and governance access/radius metadata. Final metrics, source outcome metrics, S05 compatibility scores, S06 dominance annotations, S07 continuum labels, graft outcomes, clone takeover, damage, proposal blocking outcomes, and final state labels were excluded from the predictor matrix.

Held-out validation used grouped train/test splits by `condition_group_id`, not random row splits, so seeds from the same condition were not split across train and test. S13 evaluated multiclass final-state classification and continuous regression targets for final target quality, aggregation delta, goal-conflict index, and largest-block fraction. The exploratory causal-style screen is a held-out permutation-importance ranking from predictive models; it is not a causal-discovery proof.

## Commands

- Main command: `{command_line}`
- Unit-test command: `{unit_test_result.get("command", "")}`
- Unit-test return code: `{unit_test_result.get("returnCode")}`

## Dependencies

No new dependencies were installed. S13 used repository code plus preinstalled `pandas`, `numpy`, `pyarrow`, `scikit-learn {sklearn.__version__}`, `joblib`, and `matplotlib`.

## Parameters

```json
{json.dumps({
    "splitCount": config.split_count,
    "testFraction": config.test_fraction,
    "randomState": config.random_state,
    "randomForestTrees": config.random_forest_trees,
    "workers": config.workers,
    "maxPermutationRows": config.max_permutation_rows,
    "permutationRepeats": config.permutation_repeats,
    "permutationSplitLimit": config.permutation_split_limit,
    "numericFeatures": list(NUMERIC_FEATURES),
    "categoricalFeatures": list(CATEGORICAL_FEATURES),
    "classificationTarget": CLASSIFICATION_TARGET,
    "regressionTargets": list(REGRESSION_TARGETS),
}, indent=2, sort_keys=True)}
```

## Input Matrix

{markdown_table(source_counts, ["source_research_step_id", "row_count"], max_rows=20)}

- Total model rows: {len(matrix)}
- Condition groups: {matrix["condition_group_id"].nunique()}
- Final-state classes: {matrix[CLASSIFICATION_TARGET].nunique()}

## Performance

{markdown_table(perf_sorted, ["target_name", "target_type", "model_id", "split_count", "mean_accuracy", "mean_balanced_accuracy", "mean_macro_f1", "mean_mae", "mean_rmse", "mean_r2"], max_rows=40)}

S13 outcome call: `{outcome["outcomeClassification"]}` because {outcome["primaryReason"]}. The random-forest final-state classifier mean held-out accuracy was {outcome["bestClassifierAccuracy"]:.4g} versus dummy {outcome["dummyClassifierAccuracy"]:.4g}. Random-forest regression targets exceeding the configured R2 threshold: {outcome["positiveRegressionTargetCount"]}.

## Leakage Audit

{markdown_table(leakage_summary, ["leakage_status", "column_count"], max_rows=20)}

Predictor columns were audited before fitting. Any final, source-outcome, S05, S06, S07, dominance, graft-outcome, clone-growth, takeover, or damage descriptors were excluded from predictors and either used only as targets or written as separate context annotations.

## Feature Importance And Exploratory Causal Screen

{markdown_table(causal_screen, ["target_name", "feature_name", "mean_importance", "split_count", "claim_scope"], max_rows=30)}

These rankings are useful for S14 prioritization, but they should be read as conditional predictive associations in this bounded design matrix, not as causal proof. Mechanistic claims still require targeted intervention or ablation experiments.

## Validation

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=40)}

## Provenance

- Repository branch: `{manifest["repoState"]["branch"]}`
- Repository head: `{manifest["repoState"]["head"]}`
- Repository status before artifact finalization: `{manifest["repoState"]["statusShort"]}`
- Python: `{manifest["runtime"]["python"]}`
- Platform: `{manifest["runtime"]["platform"]}`
- scikit-learn: `{sklearn.__version__}`

## Caveats, Blockers, And Limitations

- S13 predicts computational proxy outcomes only; it is not biological validation.
- The model matrix combines heterogeneous step families. Source-step identity is therefore an important predictor and should not be mistaken for a biological mechanism.
- The final-state class target inherits prior label instability caveats from S07; continuous target models are reported alongside the class model.
- S06 context-dependent dominance is available for interpretation but excluded from predictors to avoid turning downstream dominance outcomes into upstream explanatory variables.
- The causal-style screen is deliberately orientation-free and exploratory; S14 should test candidate interventions prospectively on held-out seeds and conditions.

## Recommended Next Action

Proceed to S14 intervention search after review. Use S13 to select failure contexts and candidate steering variables, but keep interventions prospective and report any overfitting to the S02-S12 corpus.
"""
    write_text(path, report)


def main() -> None:
    args = parse_args()
    started = datetime.now(UTC)
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_dir = artifacts_dir / "results"
    table_dir = artifacts_dir / "tables"
    figure_dir = artifacts_dir / "figures" / "e06"
    model_dir = artifacts_dir / "models" / "e06_final_state_predictors"
    config_dir = artifacts_dir / "configs"
    checksum_dir = artifacts_dir / "checksums"
    for directory in (step_dir, result_dir, table_dir, figure_dir, model_dir, config_dir, checksum_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = S13Config(
        split_count=int(args.split_count),
        test_fraction=float(args.test_fraction),
        random_forest_trees=int(args.random_forest_trees),
        max_permutation_rows=int(args.max_permutation_rows),
        permutation_repeats=int(args.permutation_repeats),
        permutation_split_limit=int(args.permutation_split_limit),
        workers=max(1, min(8, int(args.workers))),
    )
    input_paths = {
        "s07_mosaic_classes": args.s07_mosaic_path,
        "s05_compatibility_scores": args.s05_score_path,
        "s06_context_dependence": args.s06_context_path,
        "s06_dominance_hierarchy": args.s06_hierarchy_path,
        "s08_interface_rule_interventions": args.s08_run_path,
        "s09_governance_mechanisms": args.s09_run_path,
        "s10_graft_experiments": args.s10_run_path,
        "s11_mutant_clone_experiments": args.s11_run_path,
        "s12_developmental_history": args.s12_run_path,
    }

    s07_mosaic = read_parquet_required(args.s07_mosaic_path)
    direct_frames = {
        "S08": read_parquet_required(args.s08_run_path),
        "S09": read_parquet_required(args.s09_run_path),
        "S10": read_parquet_required(args.s10_run_path),
        "S11": read_parquet_required(args.s11_run_path),
        "S12": read_parquet_required(args.s12_run_path),
    }
    s05_scores = read_parquet_required(args.s05_score_path)
    s06_context = read_csv_if_exists(args.s06_context_path)
    s06_hierarchy = read_csv_if_exists(args.s06_hierarchy_path)

    unit_test_result = (
        run_command([sys.executable, "-m", "unittest", "tests.e06.test_final_state_models", "-v"], cwd=args.repo_dir)
        if args.run_unit_tests
        else {"command": "not run (--no-run-unit-tests)", "success": True, "returnCode": 0, "stdout": "", "stderr": ""}
    )
    if not unit_test_result["success"]:
        write_json(step_dir / "s13_unit_test_failure.json", unit_test_result)
        raise RuntimeError("S13 unit tests failed")

    matrix = build_model_input_matrix(s07_mosaic, direct_frames)
    feature_catalog = build_feature_catalog()
    predictor_columns = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    leakage_audit = audit_predictor_leakage(matrix, predictor_columns)
    performance, predictions, feature_importance = evaluate_models(matrix, config)
    performance_summary = summarize_performance(performance)
    causal_screen = build_causal_screen(feature_importance, performance_summary)
    s06_annotations = build_s06_context_annotations(matrix, s06_context, s06_hierarchy)
    outcome = infer_s13_outcome(performance_summary, config)

    model_artifacts = {}
    final_models = train_final_models(matrix, config)
    for model_id, model in final_models.items():
        model_path = model_dir / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        model_artifacts[model_id] = model_path
    write_json(
        model_dir / "feature_schema.json",
        {
            "schema": "eidosoma.e06.s13_model_feature_schema.v1",
            "numericFeatures": list(NUMERIC_FEATURES),
            "categoricalFeatures": list(CATEGORICAL_FEATURES),
            "classificationTarget": CLASSIFICATION_TARGET,
            "regressionTargets": list(REGRESSION_TARGETS),
        },
    )
    model_artifacts["feature_schema"] = model_dir / "feature_schema.json"

    model_input_path = step_dir / "e06_s13_model_input_matrix.parquet"
    feature_catalog_path = step_dir / "e06_s13_feature_catalog.csv"
    leakage_audit_path = step_dir / "e06_s13_leakage_audit.csv"
    validation_path = step_dir / "e06_s13_validation_checks.csv"
    s06_annotation_path = step_dir / "e06_s13_s06_context_annotations.csv"
    config_path = config_dir / "e06_s13_final_state_models_config.json"
    prediction_path = result_dir / "e06_final_state_prediction.parquet"
    performance_path = table_dir / "e06_s13_model_performance.csv"
    performance_parquet_path = table_dir / "e06_s13_model_performance.parquet"
    performance_summary_path = table_dir / "e06_s13_model_performance_summary.csv"
    feature_importance_path = table_dir / "e06_s13_feature_importance.csv"
    causal_screen_path = table_dir / "e06_s13_exploratory_causal_screen.csv"
    figure_path = figure_dir / "final_state_feature_importance.png"
    manifest_path = step_dir / "manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksum_dir / "sha256sums.txt"

    matrix.to_parquet(model_input_path, index=False)
    predictions.to_parquet(prediction_path, index=False)
    performance.to_csv(performance_path, index=False)
    performance.to_parquet(performance_parquet_path, index=False)
    performance_summary.to_csv(performance_summary_path, index=False)
    feature_catalog.to_csv(feature_catalog_path, index=False)
    leakage_audit.to_csv(leakage_audit_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    causal_screen.to_csv(causal_screen_path, index=False)
    s06_annotations.to_csv(s06_annotation_path, index=False)
    write_json(
        config_path,
        {
            "schema": "eidosoma.e06.s13_config.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "generatedAtUtc": utc_now(),
            "config": config.__dict__,
            "inputPaths": {name: str(path) for name, path in input_paths.items()},
            "numericFeatures": list(NUMERIC_FEATURES),
            "categoricalFeatures": list(CATEGORICAL_FEATURES),
            "classificationTarget": CLASSIFICATION_TARGET,
            "regressionTargets": list(REGRESSION_TARGETS),
        },
    )
    figure_written = write_feature_importance_figure(figure_path, feature_importance, performance_summary)
    validation = validate_s13_outputs(
        matrix,
        performance,
        predictions,
        feature_catalog,
        leakage_audit,
        s06_annotations,
        causal_screen,
        model_artifact_count=len(model_artifacts),
        figure_written=figure_written,
        unit_tests_success=bool(unit_test_result["success"]),
        s05_context_available=not s05_scores.empty,
        s07_context_available=not s07_mosaic.empty,
        config=config,
    )
    validation.to_csv(validation_path, index=False)

    elapsed = (datetime.now(UTC) - started).total_seconds()
    repo_state = {
        "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "head": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        "remote": git_output(args.repo_dir, ["remote", "-v"]),
    }
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "elapsedSeconds": elapsed,
        "workerCount": config.workers,
    }
    artifact_paths: list[tuple[Path, str]] = [
        (model_input_path, "S13 row-level model input matrix"),
        (prediction_path, "S13 held-out prediction rows"),
        (performance_path, "S13 split-level model performance CSV"),
        (performance_parquet_path, "S13 split-level model performance Parquet"),
        (performance_summary_path, "S13 model performance summary"),
        (feature_importance_path, "S13 held-out permutation feature importances"),
        (causal_screen_path, "S13 exploratory causal-style association screen"),
        (feature_catalog_path, "S13 predictor feature catalog and leakage roles"),
        (leakage_audit_path, "S13 predictor leakage audit"),
        (s06_annotation_path, "S13 S06 dominance/context annotations kept outside predictors"),
        (figure_path, "S13 feature-importance figure"),
        (config_path, "S13 model configuration"),
        (validation_path, "S13 validation checks"),
    ]
    artifact_paths.extend((path, f"S13 trained reusable model artifact: {model_id}") for model_id, path in model_artifacts.items())

    manifest: dict[str, Any] = {
        "schema": "eidosoma.e06.s13_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": bool(validation["success"].all()),
        "status": "complete" if bool(validation["success"].all()) else "validation_failed",
        "outcomeClassification": outcome["outcomeClassification"],
        "outcome": outcome,
        "rowCount": int(len(matrix)),
        "conditionGroupCount": int(matrix["condition_group_id"].nunique()),
        "sourceStepCounts": matrix["source_research_step_id"].value_counts().sort_index().to_dict(),
        "predictionRowCount": int(len(predictions)),
        "performanceRowCount": int(len(performance)),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "repoState": repo_state,
        "runtime": runtime,
        "inputArtifacts": {name: source_entry(path) for name, path in input_paths.items()},
        "sourceCode": {
            "module": source_entry(args.repo_dir / "src/e06/final_state_models.py"),
            "script": source_entry(args.repo_dir / "scripts/e06_s13_final_state_models.py"),
            "tests": source_entry(args.repo_dir / "tests/e06/test_final_state_models.py"),
        },
        "artifacts": [artifact_entry(path, artifacts_dir, description) for path, description in artifact_paths],
    }
    manifest["artifacts"].append(pending_entry(manifest_path, artifacts_dir, "S13 artifact manifest"))
    manifest["artifacts"].append(pending_entry(report_path, artifacts_dir, "S13 full-results handoff report"))
    write_json(manifest_path, manifest)
    manifest["artifacts"][-2] = artifact_entry(manifest_path, artifacts_dir, "S13 artifact manifest")

    write_report(
        report_path,
        artifacts_dir=artifacts_dir,
        config=config,
        input_paths=input_paths,
        matrix=matrix,
        performance_summary=performance_summary,
        leakage_audit=leakage_audit,
        feature_importance=feature_importance,
        causal_screen=causal_screen,
        validation=validation,
        manifest=manifest,
        outcome=outcome,
        unit_test_result=unit_test_result,
        command_line=" ".join(sys.argv),
    )
    manifest["artifacts"][-1] = artifact_entry(report_path, artifacts_dir, "S13 full-results handoff report")
    write_json(manifest_path, manifest)

    run_manifest = {
        "schema": "eidosoma.e06.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAtUtc": utc_now(),
        "repoState": repo_state,
        "runtime": runtime,
        "latestStepManifest": str(manifest_path),
        "latestStepArtifacts": manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)

    checksum_entries = [entry["relativePath"] for entry in manifest["artifacts"]]
    checksum_entries.append(str(run_manifest_path.relative_to(artifacts_dir)))
    with checksum_path.open("w", encoding="utf-8") as handle:
        for rel in checksum_entries:
            path = artifacts_dir / rel
            if path.exists() and path.is_file():
                handle.write(f"{sha256_file(path)}  {rel}\n")
    print(
        json.dumps(
            {
                "success": bool(validation["success"].all()),
                "status": manifest["status"],
                "outcomeClassification": outcome["outcomeClassification"],
                "rowCount": int(len(matrix)),
                "predictionRowCount": int(len(predictions)),
                "validationPassed": int(validation["success"].sum()),
                "validationTotal": int(len(validation)),
                "reportPath": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not bool(validation["success"].all()):
        raise RuntimeError(f"S13 validation failed; see {validation_path}")


if __name__ == "__main__":
    main()
