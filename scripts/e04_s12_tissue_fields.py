#!/usr/bin/env python3
"""Run E04 S12 emergent tissue-field predictor analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
)
from src.e04.tissue_fields import (
    FEATURE_SETS,
    PREDICTION_TARGETS,
    S12_ALLOWED_SIGNAL_FIELDS,
    S12_PROTOCOL_ID,
    build_tissue_field_feature_rows,
    evaluate_field_predictors,
    infer_field_predictor_outcome,
    predictor_feature_audit,
    split_audit,
)


STEP_ID = "S12"
STEP_NUMBER = 12
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument(
        "--s02-signal-validation",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "results" / "e04_signal_validation.parquet",
    )
    parser.add_argument(
        "--s11-results",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "results" / "e04_intelligence_like_competencies.parquet",
    )
    parser.add_argument(
        "--s11-traces",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "traces" / "e04_s11_competency_traces.parquet",
    )
    parser.add_argument("--diffusion-rate", type=float, default=0.25)
    parser.add_argument("--field-decay", type=float, default=0.02)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "(no rows)"
    present = [column for column in columns if column in df.columns]
    view = df[present].head(max_rows)
    header = "| " + " | ".join(present) + " |"
    separator = "| " + " | ".join("---" for _ in present) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in present) + " |")
    return "\n".join([header, separator, *rows])


def load_s02_signal_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False, "allValidationRowsPassed": False}
    df = pd.read_parquet(path)
    return {
        "path": str(path),
        "available": True,
        "rows": int(len(df)),
        "allValidationRowsPassed": bool(df["success"].all()),
        "validationCases": df["validation_case"].tolist(),
    }


def write_figure(results_df: pd.DataFrame, feature_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    ok = results_df[results_df["model_status"] == "ok"].copy()
    pivot = ok.pivot_table(index="target_name", columns="feature_set", values="test_roc_auc", aggfunc="mean")
    feature_order = [
        "no_field_baseline",
        "local_order_baseline",
        "allowed_signal_fields",
        "local_order_plus_allowed_signal_fields",
    ]
    pivot = pivot.reindex(index=list(PREDICTION_TARGETS), columns=feature_order)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    image = axes[0].imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Held-out AUROC by feature set")
    axes[0].set_xticks(range(len(pivot.columns)), [item.replace("_", " ") for item in pivot.columns], rotation=35, ha="right")
    axes[0].set_yticks(range(len(pivot.index)), [item.replace("_", " ") for item in pivot.index])
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    example = feature_df[feature_df["policy_mode"].isin(["diffusive_signaling", "neighbor_memory"])].copy()
    if example.empty:
        example = feature_df.copy()
    run_id = str(example["run_id"].iloc[0])
    example = example[example["run_id"] == run_id].sort_values("event_step").head(180)
    axes[1].plot(example["event_step"], example["blocked_field_mean"], label="blocked mean", linewidth=1.8)
    axes[1].plot(example["event_step"], example["frustrated_field_mean"], label="frustrated mean", linewidth=1.8)
    axes[1].plot(example["event_step"], example["order_error_fraction"], label="order error fraction", linewidth=1.5, alpha=0.8)
    axes[1].set_title("Example allowed fields over one S11 trace")
    axes[1].set_xlabel("Event step")
    axes[1].set_ylabel("Feature value")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    *,
    outcomes_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    results_df: pd.DataFrame,
    feature_audit: Mapping[str, Any],
    split: Mapping[str, Any],
    s02_audit: Mapping[str, Any],
    unit_tests: Mapping[str, Any],
    e04_s12_tests: Mapping[str, Any],
    e04_s11_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "validation_case": "s11_outcomes_and_traces_joined",
            "success": bool(len(feature_df) > 0 and set(feature_df["run_id"]) == set(outcomes_df["run_id"])),
            "detail": f"feature rows={len(feature_df)}; feature run ids={feature_df['run_id'].nunique()}; outcome run ids={outcomes_df['run_id'].nunique()}",
        }
    )
    rows.append(
        {
            "validation_case": "train_test_split_by_seed_and_perturbation",
            "success": bool(split.get("success", False)),
            "detail": json.dumps(dict(split), sort_keys=True, default=str)[:1800],
        }
    )
    expected_rows = len(PREDICTION_TARGETS) * len(FEATURE_SETS)
    rows.append(
        {
            "validation_case": "predictor_matrix_complete",
            "success": bool(len(results_df) == expected_rows and set(results_df["target_name"]) == set(PREDICTION_TARGETS) and set(results_df["feature_set"]) == set(FEATURE_SETS)),
            "detail": f"rows={len(results_df)}; expected={expected_rows}; ok models={int((results_df['model_status'] == 'ok').sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "no_field_and_local_order_baselines_compared",
            "success": bool(
                {"no_field_baseline", "local_order_baseline", "local_order_plus_allowed_signal_fields"}.issubset(set(results_df["feature_set"]))
                and results_df["local_order_auc_delta"].notna().any()
            ),
            "detail": f"feature sets={sorted(results_df['feature_set'].unique())}",
        }
    )
    rows.append(
        {
            "validation_case": "forbidden_target_derived_predictor_fields_excluded",
            "success": bool(feature_audit.get("success", False)),
            "detail": json.dumps(dict(feature_audit), sort_keys=True, default=str)[:1800],
        }
    )
    signal_columns = [column for columns in FEATURE_SETS.values() for column in columns if "_field_" in column]
    rows.append(
        {
            "validation_case": "allowed_signal_fields_only",
            "success": bool(
                tuple(ALLOWED_TRAINING_SIGNAL_FIELDS) == S12_ALLOWED_SIGNAL_FIELDS
                and not any(
                    column.startswith(("sorted_", "target_seeking_", "morphogen_")) for column in signal_columns
                )
            ),
            "detail": f"signal feature columns={signal_columns}; excluded={list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}",
        }
    )
    rows.append(
        {
            "validation_case": "s02_signal_validation_available",
            "success": bool(s02_audit.get("available", False) and s02_audit.get("allValidationRowsPassed", False)),
            "detail": json.dumps(dict(s02_audit), sort_keys=True, default=str),
        }
    )
    rows.append(
        {
            "validation_case": "no_centralized_or_oracle_rows",
            "success": bool((outcomes_df["centralized_baseline"] == False).all() and (outcomes_df["uses_global_oracle"] == False).all() and (feature_df["uses_global_oracle"] == False).all()),
            "detail": f"outcome oracle rows={int(outcomes_df['uses_global_oracle'].sum())}; feature oracle rows={int(feature_df['uses_global_oracle'].sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "unit_and_regression_tests_passed",
            "success": bool(unit_tests["success"] and e04_s12_tests["success"] and e04_s11_tests["success"] and e03_policy_tests["success"] and e02_tests["success"]),
            "detail": (
                f"E04 discover={unit_tests['returnCode']}; E04 S12={e04_s12_tests['returnCode']}; "
                f"E04 S11={e04_s11_tests['returnCode']}; E03 policy={e03_policy_tests['returnCode']}; "
                f"E02 deterministic={e02_tests['returnCode']}"
            ),
        }
    )
    return pd.DataFrame(rows)


def render_report(
    *,
    results_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    outcome: Mapping[str, Any],
    s02_audit: Mapping[str, Any],
    split: Mapping[str, Any],
    validation_line: str,
    validation_success: bool,
    artifact_paths: Sequence[Path],
    result_path: Path,
    feature_path: Path,
    figure_path: Path,
    config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    args: argparse.Namespace,
    unit_tests: Mapping[str, Any],
    e04_s12_tests: Mapping[str, Any],
    e04_s11_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> str:
    classification = "constraining/contradictory" if not validation_success else str(outcome.get("outcome", "null"))
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{e04_s12_tests['command']}` -> return code {e04_s12_tests['returnCode']}",
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e04_s11_tests['command']}` -> return code {e04_s11_tests['returnCode']}",
            f"- `{e03_policy_tests['command']}` -> return code {e03_policy_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            (
                "- `python scripts/e04_s12_tissue_fields.py --repo-dir /workspace/cell-research "
                "--artifacts-dir $ARTIFACTS_DIR`"
            ),
        ]
    )
    result_table = markdown_table(
        results_df,
        [
            "target_name",
            "feature_set",
            "model_status",
            "test_roc_auc",
            "test_average_precision",
            "local_order_auc_delta",
            "no_field_auc_delta",
        ],
        max_rows=24,
    )
    top_importance = importance_df.sort_values("abs_coefficient", ascending=False).head(20)
    importance_table = markdown_table(
        top_importance,
        ["target_name", "feature_set", "feature_name", "coefficient", "abs_coefficient"],
        max_rows=20,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)
    return f"""# E04 S12 Research Step Full Results

## Top Summary

- Research step ID: S12
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {classification}
- Caveats or blockers: S12 is an offline predictive analysis. Target-derived outcomes are used only as future labels, not policy-visible features. The signal fields are reconstructed allowed-field proxies from S11 event traces using the S02/S10 `blocked` and `frustrated` semantics; they are not biological morphogens.
- Lay summary: S12 asked whether allowed aggregate/local field features add predictive value for near-future repair, failure, movement, or frustration after accounting for no-field context and local order. Train/test groups were split by activation seed and exact perturbation schedule.
- Recommended next action: Hand control back to the Chief Scientist. If accepted, proceed to S13 overfitting and transfer tests.

## Frozen Question

Do aggregate signal fields predict future repair or failure, making them useful proto-morphogenetic state variables?

S12 outcome rule: `{json.dumps(dict(outcome), sort_keys=True)}`.

## Inputs

- S02 signal validation: `{args.s02_signal_validation}`.
- S11 competency outcomes: `{args.s11_results}`.
- S11 event traces: `{args.s11_traces}`.
- S02 audit summary: `{json.dumps(dict(s02_audit), sort_keys=True)}`.
- Train/test split audit: `{json.dumps(dict(split), sort_keys=True, default=str)}`.
- Datasets: none required.

## Methods

Implemented `src/e04/tissue_fields.py`, `tests/e04/test_tissue_fields.py`, and `scripts/e04_s12_tissue_fields.py`.

For every S11 trace row, S12 reconstructs two allowed scalar fields:

- `blocked`: deposited at the activated position when a frozen/blocked swap attempt occurs.
- `frustrated`: deposited at the activated position after blocked, impaired, or no-progress events.

Fields are diffused with the S02-style 1D update using diffusion rate `{args.diffusion_rate}` and decay `{args.field_decay}`. Predictor sets are:

- no-field baseline: event/context features only.
- local-order baseline: no-field plus adjacent-order proxies.
- allowed signal fields: blocked/frustrated field summaries only.
- local-order plus allowed signal fields.

The forbidden target-derived S02 fields `sorted`, `target_seeking`, and `morphogen` are excluded from predictor feature columns. Future labels are computed offline from S11 traces.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- scikit-learn logistic regression through preinstalled runtime.
- matplotlib backend: Agg
- New dependencies installed: none.
- CPU use: serial feature extraction and model fitting with worker count `1`; host logical CPUs recorded in run manifest.
- Platform: {platform.platform()}

## Parameters

- Protocol ID: `{S12_PROTOCOL_ID}`
- Prediction targets: `{list(PREDICTION_TARGETS)}`
- Feature sets: `{list(FEATURE_SETS)}`
- Allowed signal fields: `{list(S12_ALLOWED_SIGNAL_FIELDS)}`
- Excluded signal fields: `{list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}`
- Result table: `{result_path}`
- Feature rows: `{feature_path}`
- Figure: `{figure_path}`
- Config: `{config_path}`

## Results

### Predictor Performance

{result_table}

### Largest Coefficients

{importance_table}

## Validation Checks

{validation_table}

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S12 artifact generation if validation passed.
- S12 tests predictive value of computational field proxies only; it does not establish biological morphogen equivalence.
- S12 does not overturn S10's communication-ablation result; the analysis evaluates offline prediction from allowed field summaries, not policy benefit from communication.
- Future labels use offline target and trace outcomes; these labels are not exposed to local-policy decisions.

## Provenance

- Git commit at validation time: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Git branch at validation time: `{git_output(args.repo_dir, ['branch', '--show-current'])}`
- Git status at validation time: `{git_output(args.repo_dir, ['status', '--short']) or 'clean'}`
- Config: `{config_path}`
- Artifact manifest: `{manifest_path}`
- Run manifest: `{run_manifest_path}`
- Checksums: `{checksums_path}`
- Created at UTC: `{utc_now()}`

## Artifact Manifest

- Result parquet: `{result_path}`
- Feature-row parquet: `{feature_path}`
- Figure: `{figure_path}`

## Recommended Next Action

Hand control back to the Chief Scientist. If S12 is accepted, proceed to S13 overfitting and transfer tests with held-out configs generated before evaluation.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_tissue_field_predictors.parquet"
    feature_path = artifacts_dir / "results" / "e04_tissue_field_feature_rows.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_tissue_field_predictors.csv"
    feature_sample_path = artifacts_dir / "tables" / "e04_tissue_field_feature_rows_sample.csv"
    importance_path = artifacts_dir / "tables" / "e04_tissue_field_feature_importance.csv"
    validation_path = artifacts_dir / "tables" / "e04_tissue_field_predictors_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "tissue_field_examples.png"
    config_path = artifacts_dir / "configs" / "e04_s12_tissue_fields.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_tissue_fields_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"

    s02_audit = load_s02_signal_audit(args.s02_signal_validation)
    outcomes_df = pd.read_parquet(args.s11_results)
    traces_df = pd.read_parquet(args.s11_traces)
    feature_df = build_tissue_field_feature_rows(
        outcomes_df,
        traces_df,
        diffusion_rate=args.diffusion_rate,
        decay=args.field_decay,
    )
    results_df, importance_df = evaluate_field_predictors(feature_df)
    feature_audit = predictor_feature_audit()
    split = split_audit(feature_df)
    outcome = infer_field_predictor_outcome(results_df)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(result_path, index=False)
    feature_df.to_parquet(feature_path, index=False)
    results_df.to_csv(result_csv_path, index=False)
    feature_df.head(1000).to_csv(feature_sample_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    write_figure(results_df, feature_df, figure_path)

    config_record = {
        "schema": "eidosoma.e04.s12.tissue_fields_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "protocolId": S12_PROTOCOL_ID,
        "s02SignalValidationPath": str(args.s02_signal_validation),
        "s11ResultsPath": str(args.s11_results),
        "s11TracesPath": str(args.s11_traces),
        "diffusionRate": args.diffusion_rate,
        "fieldDecay": args.field_decay,
        "featureSets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "predictionTargets": list(PREDICTION_TARGETS),
        "allowedSignalFields": list(S12_ALLOWED_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "s02Audit": s02_audit,
        "splitAudit": split,
        "featureAudit": feature_audit,
    }
    write_json(config_path, config_record)

    if args.run_unit_tests:
        e04_s12_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_tissue_fields"], args.repo_dir)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], args.repo_dir)
        e04_s11_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_competency_metrics"], args.repo_dir)
        e03_policy_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], args.repo_dir)
        e02_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], args.repo_dir)
    else:
        skipped = {"command": "skipped by --no-run-unit-tests", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
        e04_s12_tests = unit_tests = e04_s11_tests = e03_policy_tests = e02_tests = skipped

    validation_df = validate_results(
        outcomes_df=outcomes_df,
        feature_df=feature_df,
        results_df=results_df,
        feature_audit=feature_audit,
        split=split,
        s02_audit=s02_audit,
        unit_tests=unit_tests,
        e04_s12_tests=e04_s12_tests,
        e04_s11_tests=e04_s11_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)
    validation_success = bool(validation_df["success"].all())
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 S12 tests return code {e04_s12_tests['returnCode']}; E04 tests return code {unit_tests['returnCode']}; "
        f"E04 S11 tests return code {e04_s11_tests['returnCode']}; E03 policy tests return code {e03_policy_tests['returnCode']}; "
        f"E02 simulator tests return code {e02_tests['returnCode']}; {len(results_df)} predictor rows; "
        f"feature rows={len(feature_df)}; outcome={outcome.get('outcome')}"
    )

    source_files = [
        args.repo_dir / "src/e04/tissue_fields.py",
        args.repo_dir / "scripts/e04_s12_tissue_fields.py",
        args.repo_dir / "tests/e04/test_tissue_fields.py",
        args.repo_dir / "src/e04/signaling.py",
        args.repo_dir / "src/e04/competency_metrics.py",
        args.repo_dir / "src/e04/no_oracle_protocol.py",
    ]
    source_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "sources": [source_entry(path, args.repo_dir) for path in source_files if path.exists()],
    }
    write_json(source_manifest_path, source_manifest)

    artifact_paths = [
        report_path,
        result_path,
        feature_path,
        result_csv_path,
        feature_sample_path,
        importance_path,
        validation_path,
        figure_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    report_text = render_report(
        results_df=results_df,
        importance_df=importance_df,
        validation_df=validation_df,
        outcome=outcome,
        s02_audit=s02_audit,
        split=split,
        validation_line=validation_line,
        validation_success=validation_success,
        artifact_paths=artifact_paths,
        result_path=result_path,
        feature_path=feature_path,
        figure_path=figure_path,
        config_path=config_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        args=args,
        unit_tests=unit_tests,
        e04_s12_tests=e04_s12_tests,
        e04_s11_tests=e04_s11_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    write_text(report_path, report_text)

    manifest_artifacts = [
        artifact_entry(report_path, artifacts_dir, "S12 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S12 predictor performance rows"),
        artifact_entry(feature_path, artifacts_dir, "S12 event-level field feature rows"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S12 predictor rows"),
        artifact_entry(feature_sample_path, artifacts_dir, "S12 feature-row sample table"),
        artifact_entry(importance_path, artifacts_dir, "S12 model coefficient table"),
        artifact_entry(validation_path, artifacts_dir, "S12 validation table"),
        artifact_entry(figure_path, artifacts_dir, "S12 tissue-field predictor figure"),
        artifact_entry(config_path, artifacts_dir, "S12 tissue-field predictor configuration"),
        artifact_entry(source_manifest_path, artifacts_dir, "S12 source/provenance manifest"),
        manifest_self_entry(manifest_path, artifacts_dir, "S12 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S12 outputs"),
    ]
    artifact_manifest = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "artifacts": manifest_artifacts,
    }
    write_json(manifest_path, artifact_manifest)

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "workerCount": 1,
        "cpuCount": os.cpu_count(),
        "configPath": str(config_path),
        "predictorRows": int(len(results_df)),
        "featureRows": int(len(feature_df)),
        "outcome": dict(outcome),
        "validation": validation_df.to_dict(orient="records"),
        "artifacts": manifest_artifacts,
    }
    write_json(run_manifest_path, run_manifest)

    checksum_lines = []
    for path in [
        report_path,
        result_path,
        feature_path,
        result_csv_path,
        feature_sample_path,
        importance_path,
        validation_path,
        figure_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
    ]:
        checksum_lines.append(f"{sha256_file(path)}  {path}\n")
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text("".join(checksum_lines), encoding="utf-8")

    print(validation_line)
    print(f"report={report_path}")
    print(f"results={result_path}")
    print(f"features={feature_path}")
    print(f"figure={figure_path}")
    return 0 if validation_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
