#!/usr/bin/env python3
"""Run E07 S13 substrate-transfer tests with direct morphospace replay."""

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

from morphospace2d import audit_recovery_policy_payload
from platonic_space.substrate_transfer import (
    DEFAULT_TRANSFER_SEEDS,
    SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
    SUBSTRATE_TRANSFER_MODEL_VERSION,
    SUBSTRATE_TRANSFER_SCHEMA_VERSION,
    annotate_s08_transfer_distances,
    build_transfer_policy_panel,
    build_transfer_target_panel,
    dataframe_json_columns,
    distance_prediction_correlations,
    run_transfer_panel,
    summarize_transfer_runs,
    transfer_mapping_contract,
    transfer_outcome_classification,
    validation_checks,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S13"
STEP_NUMBER = 13
STEP_TITLE = "Test substrate transfer"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_S12_DESIGNS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S12" / "designed_policies.parquet"
DEFAULT_S12_NEAREST_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S12" / "nearest_existing_policy_comparison.parquet"
DEFAULT_S08_DISTANCES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S08" / "platonic_distances.parquet"
DEFAULT_S08_ENTITIES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S08" / "distance_entities.parquet"
FOCUSED_TESTS = ["tests.test_e07_substrate_transfer"]


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
    for step in [f"S{index:02d}" for index in range(1, 13)]:
        path = artifacts_dir / "research_steps" / step / "status.json"
        status = load_status(path)
        rows.append({"stepId": step, "path": str(path), "success": bool(status.get("success")), "status": status.get("status", "missing")})
    failed = [row for row in rows if not row["success"]]
    if failed:
        raise RuntimeError(f"S13 requires successful S01-S12 statuses; failed or missing: {failed}")
    return rows


def plot_transfer_retention(summary: pd.DataFrame, path_png: Path, path_svg: Path) -> list[Path]:
    if summary.empty:
        return []
    plot_df = summary[summary["transferTargetRole"].eq("transfer_holdout")].copy()
    plot_df["policyClass"] = np.where(
        plot_df["mappingKind"].eq("s12_designed_policy_transfer"),
        "S12 transfer",
        np.where(plot_df["mappingKind"].str.contains("random|no_transfer", regex=True), "control", "baseline"),
    )
    agg = (
        plot_df.groupby(["target_id", "policyClass"], dropna=False)["competenceRetentionRatio"]
        .median()
        .reset_index()
        .sort_values(["target_id", "policyClass"], kind="mergesort")
    )
    targets = list(dict.fromkeys(agg["target_id"].astype(str)))
    classes = ["S12 transfer", "baseline", "control"]
    colors = {"S12 transfer": "#1f7a5a", "baseline": "#2f5f8f", "control": "#8b6f2a"}
    x = np.arange(len(targets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for offset, policy_class in zip((-width, 0, width), classes, strict=True):
        values = []
        for target in targets:
            subset = agg[agg["target_id"].astype(str).eq(target) & agg["policyClass"].eq(policy_class)]
            values.append(float(subset["competenceRetentionRatio"].iloc[0]) if not subset.empty else np.nan)
        ax.bar(x + offset, values, width=width, color=colors[policy_class], label=policy_class)
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=25, ha="right")
    ax.set_ylabel("median retention ratio")
    ax.set_title("S13 substrate transfer retention")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def plot_distance_prediction(distance_summary: pd.DataFrame, path_png: Path, path_svg: Path) -> list[Path]:
    plot_df = distance_summary[
        distance_summary["transferTargetRole"].eq("transfer_holdout")
        & distance_summary["s08CompositeTransferDistance"].notna()
        & distance_summary["competenceRetentionRatio"].notna()
    ].copy()
    if plot_df.empty:
        return []
    plot_df["policyClass"] = np.where(
        plot_df["mappingKind"].eq("s12_designed_policy_transfer"),
        "S12 transfer",
        np.where(plot_df["mappingKind"].str.contains("random|no_transfer", regex=True), "control", "baseline"),
    )
    colors = {"S12 transfer": "#1f7a5a", "baseline": "#2f5f8f", "control": "#8b6f2a"}
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for policy_class, subset in plot_df.groupby("policyClass"):
        ax.scatter(
            subset["s08CompositeTransferDistance"],
            subset["competenceRetentionRatio"],
            s=42,
            alpha=0.75,
            color=colors.get(policy_class, "#64748b"),
            label=policy_class,
        )
    ax.set_xlabel("S08 composite transfer distance")
    ax.set_ylabel("competence retention ratio")
    ax.set_title("S08 distance vs transfer retention")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def write_mapping_report(path: Path, status: Mapping[str, Any], mapping_catalog: pd.DataFrame, target_catalog: pd.DataFrame) -> None:
    mapping_cols = [
        "mappingKind",
        "sourcePolicyId",
        "transferPolicyId",
        "sourceDesignProfileId",
        "analoguePolicyFamily",
        "rankWeight",
        "axisWeight",
        "affinityWeight",
        "mappingCaveat",
    ]
    target_cols = ["targetId", "transferTargetRole", "substrateClass", "nodeCount", "maxSteps", "mappingCaveat"]
    lines = [
        f"# {STEP_ID} Transfer Mapping Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; exact paths and hashes are in `artifact_manifest.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Mapping Contract",
        "",
        "```json",
        json.dumps(transfer_mapping_contract(), indent=2, sort_keys=True),
        "```",
        "",
        "## Policy Mappings",
        "",
        dataframe_markdown(mapping_catalog[[column for column in mapping_cols if column in mapping_catalog.columns]], limit=60),
        "",
        "## Target Mappings",
        "",
        dataframe_markdown(target_catalog[[column for column in target_cols if column in target_catalog.columns]], limit=20),
        "",
        "## Claim Boundary",
        "",
        SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(
    path: Path,
    status: Mapping[str, Any],
    checks: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    failed = checks[(checks["severity"].eq("error")) & (~checks["success"])] if not checks.empty else pd.DataFrame()
    summary_cols = [
        "policy_id",
        "mappingKind",
        "target_id",
        "transferRelativeErrorReductionMean",
        "competenceRetentionRatio",
        "beatsRandomOrNoTransferControl",
        "beatsBubbleLikeBaseline",
        "failureMode",
    ]
    corr_cols = ["predictor", "outcome", "n", "spearmanR", "spearmanP", "status", "interpretation"]
    lines = [
        f"# {STEP_ID} Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; exact paths and hashes are in `artifact_manifest.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Validation Checks",
        "",
        dataframe_markdown(checks, limit=60),
        "",
        "## Hard Failures",
        "",
        "None." if failed.empty else dataframe_markdown(failed, limit=20),
        "",
        "## Transfer Summary Snapshot",
        "",
        dataframe_markdown(summary[[column for column in summary_cols if column in summary.columns]], limit=40),
        "",
        "## S08 Distance Diagnostics",
        "",
        dataframe_markdown(correlations[[column for column in corr_cols if column in correlations.columns]], limit=40),
        "",
        "## Claim Boundary",
        "",
        SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status: Mapping[str, Any], summary: pd.DataFrame, correlations: pd.DataFrame, outcome: str) -> None:
    candidate = summary[
        summary["mappingKind"].eq("s12_designed_policy_transfer")
        & summary["transferTargetRole"].eq("transfer_holdout")
    ].copy()
    candidate_retention = pd.to_numeric(candidate["competenceRetentionRatio"], errors="coerce")
    beats_random = candidate["beatsRandomOrNoTransferControl"].astype(bool).mean() if not candidate.empty else float("nan")
    beats_bubble = candidate["beatsBubbleLikeBaseline"].astype(bool).mean() if not candidate.empty else float("nan")
    corr_row = correlations[
        correlations["predictor"].eq("s08CompositeTransferDistance")
        & correlations["outcome"].eq("competenceRetentionRatio")
    ]
    corr_text = "not computable"
    if not corr_row.empty and corr_row.iloc[0]["status"] == "computed":
        corr_text = f"Spearman r={float(corr_row.iloc[0]['spearmanR']):.3f}, p={float(corr_row.iloc[0]['spearmanP']):.3g}"
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
        (
            "S13 transferred the nine S12 designed DSL policies into row, 2D-grid, and irregular-graph recovery tasks through "
            "explicit local-swap analogues, then compared them against E05 baselines, randomized transfer controls, a random-swap null, "
            "and a wait-only no-transfer control."
        ),
        "",
        f"Median S12 transfer retention across held-out non-row targets: {candidate_retention.dropna().median():.3f}.",
        f"Fraction of S12 policy-target summaries beating random/no-transfer controls: {beats_random:.3f}.",
        f"Fraction beating the Bubble-like scalar-rank baseline: {beats_bubble:.3f}.",
        f"S08 composite distance versus retention: {corr_text}.",
        "",
        "These are empirical computational distances and simulator competence proxies. Unsupported exact homeostatic and chimera-growth mappings were documented but not executed.",
        "",
        "## Transfer Snapshot",
        "",
        dataframe_markdown(
            candidate[
                [
                    "policy_id",
                    "sourceDesignProfileId",
                    "target_id",
                    "transferRelativeErrorReductionMean",
                    "competenceRetentionRatio",
                    "beatsRandomOrNoTransferControl",
                    "beatsBubbleLikeBaseline",
                    "failureMode",
                ]
            ].head(30),
            limit=30,
        ),
        "",
        "## Claim Boundary",
        "",
        SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--s12-designs", type=Path, default=DEFAULT_S12_DESIGNS_PATH)
    parser.add_argument("--s12-nearest", type=Path, default=DEFAULT_S12_NEAREST_PATH)
    parser.add_argument("--s08-distances", type=Path, default=DEFAULT_S08_DISTANCES_PATH)
    parser.add_argument("--s08-entities", type=Path, default=DEFAULT_S08_ENTITIES_PATH)
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_TRANSFER_SEEDS))
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures"
    model_dir = artifacts_dir / "models" / "e07_substrate_transfer"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    upstream_statuses = require_upstream_statuses(artifacts_dir)
    missing_inputs = [str(path) for path in (args.s12_designs, args.s12_nearest, args.s08_distances, args.s08_entities) if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"S13 missing required upstream inputs: {missing_inputs}")

    seeds = DEFAULT_TRANSFER_SEEDS[: max(1, int(args.seed_count))]
    designed = pd.read_parquet(args.s12_designs)
    nearest = pd.read_parquet(args.s12_nearest)
    s08_distances = pd.read_parquet(args.s08_distances)
    s08_entities = pd.read_parquet(args.s08_entities)

    policy_panel = build_transfer_policy_panel(designed, nearest, include_randomized_controls=True)
    target_panel = build_transfer_target_panel()
    run_rows, trace_rows, severity_rows = run_transfer_panel(policy_panel.policies, target_panel.targets, seeds=seeds)
    transfer_summary = summarize_transfer_runs(run_rows, policy_panel.mapping_catalog, target_panel.target_catalog)
    distance_summary = annotate_s08_transfer_distances(transfer_summary, s08_distances)
    distance_correlations = distance_prediction_correlations(distance_summary)
    policy_audits = pd.DataFrame(audit_recovery_policy_payload(policy) for policy in policy_panel.policies)
    checks = validation_checks(
        designed_policies=designed,
        policy_panel=policy_panel,
        target_panel=target_panel,
        run_rows=run_rows,
        transfer_summary=distance_summary,
        distance_correlations=distance_correlations,
        seeds=seeds,
        upstream_statuses=upstream_statuses,
    )
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
    outcome = transfer_outcome_classification(distance_summary, distance_correlations)
    candidate = distance_summary[
        distance_summary["mappingKind"].eq("s12_designed_policy_transfer")
        & distance_summary["transferTargetRole"].eq("transfer_holdout")
    ].copy()
    candidate_retention = pd.to_numeric(candidate["competenceRetentionRatio"], errors="coerce")
    validation_result = (
        f"passed: {len(policy_panel.policies)} executed policy/control specs, {len(target_panel.targets)} targets, "
        f"{len(run_rows)} direct transfer runs across {len(seeds)} held-out seeds, "
        f"median S12 held-out retention {candidate_retention.dropna().median():.3f}, "
        f"{len(hard_failures)} hard validation failures"
        if hard_failures.empty
        else f"failed: {len(hard_failures)} hard validation failures"
    )
    caveats = (
        "S12 policies are executed through documented E05 local-swap analogues rather than native S12 DSL on every substrate; "
        "several S12 designs are near or identical to Bubble-like references; exact homeostatic birth/death and chimera-growth mappings are unsupported; "
        "S08 distances are empirical proxies and graph target distance components are missing."
    )
    recommended_next = "Chief Scientist review S13 transfer mappings, retention/failure modes, and S08 distance diagnostics before authorizing S14."
    status_state = "completed" if hard_failures.empty else "completed_with_validation_errors"

    artifact_paths: list[Path] = []
    artifact_paths += write_dataframe(policy_panel.mapping_catalog, step_dir / "transfer_mapping_catalog")
    artifact_paths += write_dataframe(target_panel.target_catalog, step_dir / "transfer_target_catalog")
    artifact_paths += write_dataframe(policy_audits, step_dir / "policy_payload_audits")
    artifact_paths += write_dataframe(run_rows, step_dir / "direct_transfer_runs")
    artifact_paths += write_dataframe(trace_rows, step_dir / "transfer_trace_snapshots", csv=False)
    artifact_paths += write_dataframe(severity_rows, step_dir / "scramble_severity_records", csv=False)
    artifact_paths += write_dataframe(transfer_summary, step_dir / "transfer_metric_summary")
    artifact_paths += write_dataframe(distance_summary, step_dir / "s08_distance_transfer_prediction")
    artifact_paths += write_dataframe(distance_correlations, step_dir / "s08_distance_correlation")
    artifact_paths += write_dataframe(checks, step_dir / "validation_checks")
    artifact_paths += write_dataframe(run_rows, results_dir / "e07_substrate_transfer")
    artifact_paths += write_dataframe(distance_summary, results_dir / "e07_substrate_transfer_summary")
    artifact_paths += plot_transfer_retention(
        distance_summary,
        figures_dir / "e07_s13_transfer_retention.png",
        figures_dir / "e07_s13_transfer_retention.svg",
    )
    artifact_paths += plot_distance_prediction(
        distance_summary,
        figures_dir / "e07_s13_s08_distance_prediction.png",
        figures_dir / "e07_s13_s08_distance_prediction.svg",
    )

    transfer_card = {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "modelVersion": SUBSTRATE_TRANSFER_MODEL_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "createdUtc": utc_now(),
        "sourceArtifacts": {
            "s12DesignedPolicies": str(args.s12_designs),
            "s12NearestExisting": str(args.s12_nearest),
            "s08Distances": str(args.s08_distances),
            "s08Entities": str(args.s08_entities),
        },
        "mappingContract": transfer_mapping_contract(),
        "targetCount": int(len(target_panel.targets)),
        "executedPolicyCount": int(len(policy_panel.policies)),
        "directRunCount": int(len(run_rows)),
        "seeds": list(seeds),
        "serialWorkerCount": 1,
        "threadEnvironment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "upstreamRows": {
            "s12DesignedPolicyRows": int(len(designed)),
            "s12NearestExistingRows": int(len(nearest)),
            "s08DistanceRows": int(len(s08_distances)),
            "s08EntityRows": int(len(s08_entities)),
        },
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "claimBoundary": SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
    }
    transfer_card_path = model_dir / "transfer_card.json"
    write_json(transfer_card_path, transfer_card)
    artifact_paths.append(transfer_card_path)

    contract_path = step_dir / "transfer_mapping_contract.json"
    write_json(contract_path, transfer_mapping_contract())
    artifact_paths.append(contract_path)

    test_log_path = step_dir / "focused_tests.json"
    if test_result is not None:
        write_json(test_log_path, test_result)
        artifact_paths.append(test_log_path)

    summary_path = step_dir / "summary.md"
    validation_report_path = step_dir / "validation_report.md"
    mapping_report_path = step_dir / "transfer_mapping_report.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    artifact_paths += [summary_path, validation_report_path, mapping_report_path, status_path, manifest_path]

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
        "directTransferRunCount": int(len(run_rows)),
        "executedPolicyCount": int(len(policy_panel.policies)),
        "targetCount": int(len(target_panel.targets)),
        "seedCount": int(len(seeds)),
        "s12DesignedPolicyCount": int(len(designed)),
        "medianS12HeldoutRetention": float(candidate_retention.dropna().median()) if candidate_retention.notna().any() else None,
        "upstreamStatusSummary": upstream_statuses,
        "repositoryCodePaths": [
            "platonic_space/substrate_transfer.py",
            "scripts/e07_s13_substrate_transfer.py",
            "tests/test_e07_substrate_transfer.py",
        ],
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "statusShort": git_value(["status", "--short"]),
            "remote": git_value(["remote", "get-url", "origin"]),
        },
        "claimBoundary": SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
    }

    write_summary(summary_path, status, distance_summary, distance_correlations, outcome)
    write_validation_report(validation_report_path, status, checks, distance_summary, distance_correlations)
    write_mapping_report(mapping_report_path, status, policy_panel.mapping_catalog, target_panel.target_catalog)
    write_json(status_path, status)
    manifest = {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdUtc": utc_now(),
        "artifacts": collect_artifacts(artifact_paths),
        "repositoryCodePaths": status["repositoryCodePaths"],
        "git": status["git"],
        "claimBoundary": SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
    }
    write_json(manifest_path, manifest)

    print(json.dumps({key: status[key] for key in ("researchStepId", "success", "status", "validationResult", "outcomeClassification")}, indent=2))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

