#!/usr/bin/env python3
"""Run E07 S12 inverse design with held-out direct simulation validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
for thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from platonic_space.inverse_design import (
    INVERSE_DESIGN_CLAIM_BOUNDARY,
    INVERSE_DESIGN_MODEL_VERSION,
    INVERSE_DESIGN_SCHEMA_VERSION,
    build_inverse_design_target_profiles,
    dataframe_json_columns,
    generate_inverse_design_pool,
    holdout_panel,
    nearest_existing_policy_comparison,
    policy_outcome_classification,
    reference_policy_table,
    run_design_panel,
    s11_support_table,
    score_candidate_pool,
    select_designed_policies,
    summarize_profile_validation,
    training_panel,
    validation_checks,
    write_designed_policy_files,
)
from platonic_space.world_schema import compact_json, sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S12"
STEP_NUMBER = 12
STEP_TITLE = "Do inverse design"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_S11_TARGETS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S11" / "target_performance_summary.parquet"
DEFAULT_S11_DIRECT_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S11" / "direct_validation_results.parquet"
DEFAULT_S04_CORPUS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S04" / "unified_behavior_corpus.parquet"
DEFAULT_S05_ERROR_LIMITS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "error_limits.parquet"
DEFAULT_S06_POLICIES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "policy_behavior_profiles.parquet"
DEFAULT_S07_GOALS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S07" / "goal_behavior_profiles.parquet"
FOCUSED_TESTS = ["tests.test_e07_inverse_design"]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def run_command(command: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return {
        "command": list(command),
        "cwd": str(cwd),
        "returncode": int(completed.returncode),
        "success": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsedSeconds": round(time.perf_counter() - started, 6),
    }


def git_value(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def write_dataframe(df: pd.DataFrame, stem: Path, csv: bool = True) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    out = dataframe_json_columns(df)
    if csv:
        csv_path = stem.with_suffix(".csv")
        out.to_csv(csv_path, index=False)
        paths.append(csv_path)
    parquet_path = stem.with_suffix(".parquet")
    out.to_parquet(parquet_path, index=False)
    paths.append(parquet_path)
    return paths


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(out, key=lambda row: row["path"])


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ").replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def dataframe_markdown(df: pd.DataFrame, limit: int = 40) -> str:
    if df.empty:
        return "No rows."
    display = df.head(limit).copy()
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"success": False, "missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def require_upstream_statuses(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in [f"S{index:02d}" for index in range(1, 12)]:
        path = artifacts_dir / "research_steps" / step / "status.json"
        status = load_status(path)
        rows.append({"stepId": step, "path": str(path), "success": bool(status.get("success")), "status": status.get("status", "missing")})
    failed = [row for row in rows if not row["success"]]
    if failed:
        raise RuntimeError(f"S12 requires successful S01-S11 statuses; failed or missing: {failed}")
    return rows


def plot_selection_surface(scored: pd.DataFrame, designs: pd.DataFrame, path_png: Path, path_svg: Path) -> list[Path]:
    if scored.empty:
        return []
    selected = set(designs["policyId"].astype(str)) if not designs.empty else set()
    profiles = list(dict.fromkeys(scored["designProfileId"].astype(str)))
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), squeeze=False, constrained_layout=True)
    for ax, profile_id in zip(axes.ravel(), profiles):
        subset = scored[scored["designProfileId"].astype(str).eq(profile_id)]
        ax.scatter(subset["targetProfileDistance"], subset["profileScore"], s=22, alpha=0.45, color="#64748b", label="pool")
        selected_subset = subset[subset["policyId"].astype(str).isin(selected)]
        if not selected_subset.empty:
            ax.scatter(
                selected_subset["targetProfileDistance"],
                selected_subset["profileScore"],
                s=60,
                color="#1f7a5a",
                edgecolor="black",
                linewidth=0.5,
                label="selected",
            )
        ax.set_title(profile_id)
        ax.set_xlabel("target distance")
        ax.set_ylabel("training profile score")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def plot_validation_summary(profile_summary: pd.DataFrame, path_png: Path, path_svg: Path) -> list[Path]:
    if profile_summary.empty:
        return []
    plot_df = profile_summary.copy()
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(x - 0.18, plot_df["bestHeldoutCompositeScore"], width=0.36, color="#2f5f8f", label="best composite")
    ax.bar(x + 0.18, plot_df["validatedDesignCount"], width=0.36, color="#8b6f2a", label="validated count")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["profileId"], rotation=20, ha="right")
    ax.set_ylim(0, max(1.05, float(pd.to_numeric(plot_df["validatedDesignCount"], errors="coerce").max()) + 0.5))
    ax.set_ylabel("score or count")
    ax.set_title("S12 held-out profile validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def write_inverse_design_report(
    path: Path,
    status: Mapping[str, Any],
    support: pd.DataFrame,
    profiles: pd.DataFrame,
    designs: pd.DataFrame,
) -> None:
    profile_cols = ["profileId", "profileName", "s11SupportFamiliesJson", "deferredOrSecondaryFamiliesJson", "caveat"]
    design_cols = [
        "policyId",
        "designProfileId",
        "designRankWithinProfile",
        "profileScore",
        "targetProfileDistance",
        "isExactDuplicateOfExisting",
        "designCaveat",
    ]
    lines = [
        f"# {STEP_ID} Inverse Design Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; primary target, pool, design, validation, policy DSL, result, figure, status, and manifest artifacts are listed in `status.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## S11 Evidence Decisions",
        "",
        dataframe_markdown(support, limit=30),
        "",
        "## Target Profiles",
        "",
        dataframe_markdown(profiles[[column for column in profile_cols if column in profiles.columns]], limit=10),
        "",
        "## Selected Designed Policies",
        "",
        dataframe_markdown(designs[[column for column in design_cols if column in designs.columns]], limit=20),
        "",
        "## Claim Boundary",
        "",
        INVERSE_DESIGN_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(
    path: Path,
    status: Mapping[str, Any],
    design_validation: pd.DataFrame,
    profile_summary: pd.DataFrame,
    nearest: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    validation_cols = [
        "policyId",
        "designProfileId",
        "designRankWithinProfile",
        "heldoutCompositeScore",
        "heldoutTargetProfileDistance",
        "profileValidated",
        "validationDecision",
    ]
    nearest_cols = ["policyId", "nearestExistingPolicyId", "behaviorDistanceOnHeldoutPanel", "exactBehaviorMatchOnPanel"]
    failed = checks[(checks["severity"].eq("error")) & (~checks["success"])] if not checks.empty else pd.DataFrame()
    lines = [
        f"# {STEP_ID} Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; see `artifact_manifest.json` for hashes.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Profile Summary",
        "",
        dataframe_markdown(profile_summary, limit=20),
        "",
        "## Designed Policy Validation",
        "",
        dataframe_markdown(design_validation[[column for column in validation_cols if column in design_validation.columns]], limit=40),
        "",
        "## Nearest Existing Policies",
        "",
        dataframe_markdown(nearest[[column for column in nearest_cols if column in nearest.columns]], limit=40),
        "",
        "## Validation Checks",
        "",
        dataframe_markdown(checks, limit=40),
        "",
        "## Hard Failures",
        "",
        "None." if failed.empty else dataframe_markdown(failed, limit=20),
        "",
        "## Claim Boundary",
        "",
        INVERSE_DESIGN_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status: Mapping[str, Any], profile_summary: pd.DataFrame, outcome: str) -> None:
    lines = [
        f"# {STEP_ID} Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; status and manifest enumerate exact paths.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Outcome classification: {outcome}.",
        "",
        "S12 translated S11 replay-supported targets into three direct simulator profiles: balanced sorting/low activation cost, stuck-frozen robustness, and candidate/null chimera aggregation. Sparse or high-scale S11 targets, including error reduction, compatibility, repair success, and surrogate low swap count, were excluded as primary design objectives or treated only as direct measured caveats.",
        "",
        "Held-out validation used independent scheduler/tie-breaker seeds and direct CPU simulations. The policy outputs are executable DSL JSON artifacts, not biological designs.",
        "",
        "## Profile Results",
        "",
        dataframe_markdown(profile_summary, limit=20),
        "",
        "## Claim Boundary",
        "",
        INVERSE_DESIGN_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--s11-targets", type=Path, default=DEFAULT_S11_TARGETS_PATH)
    parser.add_argument("--s11-direct", type=Path, default=DEFAULT_S11_DIRECT_PATH)
    parser.add_argument("--per-profile", type=int, default=3)
    parser.add_argument("--reference-size", type=int, default=48)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures"
    model_dir = artifacts_dir / "models" / "e07_inverse_design"
    policy_dir = step_dir / "designed_policy_dsl"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    upstream_statuses = require_upstream_statuses(artifacts_dir)
    missing_inputs = [str(path) for path in (args.s11_targets, args.s11_direct) if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"S12 missing required S11 inputs: {missing_inputs}")

    s11_targets = pd.read_parquet(args.s11_targets)
    s11_direct = pd.read_parquet(args.s11_direct)
    s04_rows = int(pd.read_parquet(DEFAULT_S04_CORPUS_PATH).shape[0]) if DEFAULT_S04_CORPUS_PATH.exists() else None
    s05_error_limits_rows = int(pd.read_parquet(DEFAULT_S05_ERROR_LIMITS_PATH).shape[0]) if DEFAULT_S05_ERROR_LIMITS_PATH.exists() else None

    support = s11_support_table(s11_targets, s11_direct)
    profiles = build_inverse_design_target_profiles(s11_targets, s11_direct)
    pool = generate_inverse_design_pool(reference_size=512)
    valid_pool = pool[pool["poolStatus"].astype(str).eq("valid")].copy()

    train_runs, train_vectors, _train_traces = run_design_panel(valid_pool, training_panel(), include_traces=False)
    scored, train_metric_summary = score_candidate_pool(valid_pool, train_vectors, profiles)
    designs = select_designed_policies(scored, per_profile=int(args.per_profile))
    policy_files = write_designed_policy_files(designs, policy_dir)

    holdout_runs, holdout_vectors, holdout_traces = run_design_panel(designs, holdout_panel(), include_traces=True)
    design_validation, profile_summary = summarize_profile_validation(designs, holdout_vectors, profiles)

    references = reference_policy_table(reference_size=int(args.reference_size))
    reference_runs, reference_vectors, _reference_traces = run_design_panel(references, holdout_panel(), include_traces=False)
    nearest_vectors = pd.concat([holdout_vectors, reference_vectors], ignore_index=True, sort=False)
    nearest = nearest_existing_policy_comparison(design_validation, nearest_vectors)
    design_validation = design_validation.merge(nearest, on=["policyId", "designProfileId"], how="left")
    result_table = design_validation.merge(
        policy_files[["policyId", "dslPath", "roundtripSuccess"]],
        on="policyId",
        how="left",
    )

    checks = validation_checks(profiles, pool, designs, holdout_runs, design_validation, policy_files)
    test_result = None
    if not args.skip_tests:
        test_result = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS])
        checks = pd.concat(
            [
                checks,
                pd.DataFrame(
                    [
                        {
                            "checkId": "focused_unit_tests",
                            "severity": "error",
                            "success": bool(test_result["success"]),
                            "observed": f"returncode={test_result['returncode']}",
                            "expected": "returncode=0",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    hard_failures = checks[checks["severity"].eq("error") & ~checks["success"]]
    outcome = policy_outcome_classification(profile_summary)
    validation_result = (
        f"passed: {len(designs)} designed policies, {len(holdout_runs)} held-out direct simulation runs, "
        f"{int(design_validation['profileValidated'].sum()) if not design_validation.empty else 0} profile-validating designs, "
        f"{int(profile_summary['validatedDesignCount'].astype(int).gt(0).sum()) if not profile_summary.empty else 0} target profiles with at least one validation; "
        f"{len(hard_failures)} hard validation failures"
        if hard_failures.empty
        else f"failed: {len(hard_failures)} hard validation failures"
    )
    caveats = (
        "Computational DSL/simulator inverse design only; S11 sparse/high-scale targets were avoided or caveated; nearest-existing "
        "behavior distances are panel-local; stuck-frozen tasks can be physically unsortable; no causal or biological claims."
    )
    recommended_next = "Chief Scientist review S12 inverse-design artifacts, target exclusions, and direct-validation caveats before authorizing S13."
    status_state = "completed" if hard_failures.empty else "completed_with_validation_errors"

    artifact_paths: list[Path] = []
    artifact_paths += write_dataframe(support, step_dir / "s11_target_support_decisions")
    artifact_paths += write_dataframe(profiles, step_dir / "inverse_design_target_profiles")
    artifact_paths += write_dataframe(pool, step_dir / "inverse_design_candidate_pool")
    artifact_paths += write_dataframe(train_runs, step_dir / "training_simulation_runs", csv=False)
    artifact_paths += write_dataframe(train_vectors, step_dir / "training_competence_vectors", csv=False)
    artifact_paths += write_dataframe(train_metric_summary, step_dir / "training_policy_metric_summary")
    artifact_paths += write_dataframe(scored, step_dir / "candidate_profile_scores")
    artifact_paths += write_dataframe(designs, step_dir / "designed_policies")
    artifact_paths += write_dataframe(policy_files, step_dir / "designed_policy_file_index")
    artifact_paths += write_dataframe(holdout_runs, step_dir / "holdout_simulation_runs")
    artifact_paths += write_dataframe(holdout_vectors, step_dir / "holdout_competence_vectors", csv=False)
    artifact_paths += write_dataframe(holdout_traces, step_dir / "holdout_trace_summaries", csv=False)
    artifact_paths += write_dataframe(references, step_dir / "nearest_existing_reference_policies")
    artifact_paths += write_dataframe(reference_runs, step_dir / "nearest_existing_reference_runs", csv=False)
    artifact_paths += write_dataframe(reference_vectors, step_dir / "nearest_existing_reference_vectors", csv=False)
    artifact_paths += write_dataframe(nearest, step_dir / "nearest_existing_policy_comparison")
    artifact_paths += write_dataframe(design_validation, step_dir / "inverse_design_validation")
    artifact_paths += write_dataframe(profile_summary, step_dir / "profile_validation_summary")
    artifact_paths += write_dataframe(checks, step_dir / "validation_checks")
    artifact_paths += write_dataframe(result_table, results_dir / "e07_inverse_design")
    artifact_paths += write_dataframe(holdout_runs, results_dir / "e07_inverse_design_holdout_runs", csv=False)
    artifact_paths += plot_selection_surface(
        scored,
        designs,
        figures_dir / "e07_s12_inverse_design_profile_scores.png",
        figures_dir / "e07_s12_inverse_design_profile_scores.svg",
    )
    artifact_paths += plot_validation_summary(
        profile_summary,
        figures_dir / "e07_s12_inverse_design_validation.png",
        figures_dir / "e07_s12_inverse_design_validation.svg",
    )

    model_card = {
        "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
        "modelVersion": INVERSE_DESIGN_MODEL_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "createdUtc": utc_now(),
        "candidateGeneration": "Deterministic DSL mutation/search over local adjacent-swap policies.",
        "selectionMethod": "Training-panel direct simulation scored against S11-supported target profiles.",
        "heldoutValidation": "Independent scheduler/tie-breaker seeds over sorting, duplicates, stuck-frozen, transfer, and candidate/null chimera tasks.",
        "targetProfiles": dataframe_json_columns(profiles).to_dict(orient="records"),
        "trainingPanel": [task.to_dict() for task in training_panel().tasks],
        "holdoutPanel": [task.to_dict() for task in holdout_panel().tasks],
        "serialWorkerCount": 1,
        "threadEnvironment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "upstreamRows": {
            "s04UnifiedBehaviorCorpusRows": s04_rows,
            "s05ErrorLimitRows": s05_error_limits_rows,
            "s11TargetPerformanceRows": int(len(s11_targets)),
            "s11DirectValidationRows": int(len(s11_direct)),
        },
        "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
    }
    model_card_path = model_dir / "inverse_design_card.json"
    write_json(model_card_path, model_card)
    artifact_paths.append(model_card_path)

    test_log_path = step_dir / "focused_tests.json"
    if test_result is not None:
        write_json(test_log_path, test_result)
        artifact_paths.append(test_log_path)

    summary_path = step_dir / "summary.md"
    validation_report_path = step_dir / "validation_report.md"
    design_report_path = step_dir / "inverse_design_report.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    artifact_paths += [summary_path, validation_report_path, design_report_path, status_path, manifest_path]

    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(hard_failures.empty),
        "status": status_state,
        "artifactsWritten": [str(path) for path in artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next,
        "experimentId": EXPERIMENT_ID,
        "title": STEP_TITLE,
        "outcomeClassification": outcome,
        "completedUtc": utc_now(),
        "runtimeSeconds": round(time.perf_counter() - started, 3),
        "designedPolicyCount": int(len(designs)),
        "heldoutDirectRunCount": int(len(holdout_runs)),
        "validatedDesignCount": int(design_validation["profileValidated"].sum()) if not design_validation.empty else 0,
        "validatedProfileCount": int(profile_summary["validatedDesignCount"].astype(int).gt(0).sum()) if not profile_summary.empty else 0,
        "upstreamStatusSummary": upstream_statuses,
        "repositoryCodePaths": [
            "platonic_space/inverse_design.py",
            "scripts/e07_s12_inverse_design.py",
            "tests/test_e07_inverse_design.py",
        ],
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "statusShort": git_value(["status", "--short"]),
            "remote": git_value(["remote", "get-url", "origin"]),
        },
        "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
    }

    write_summary(summary_path, status, profile_summary, outcome)
    write_validation_report(validation_report_path, status, design_validation, profile_summary, nearest, checks)
    write_inverse_design_report(design_report_path, status, support, profiles, designs)
    write_json(status_path, status)
    manifest = {
        "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdUtc": utc_now(),
        "artifacts": collect_artifacts(artifact_paths),
        "repositoryCodePaths": status["repositoryCodePaths"],
        "git": status["git"],
        "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
    }
    write_json(manifest_path, manifest)

    print(json.dumps({key: status[key] for key in ("researchStepId", "success", "status", "validationResult", "outcomeClassification")}, indent=2))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
