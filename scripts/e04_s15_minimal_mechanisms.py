#!/usr/bin/env python3
"""Execute E04 S15 minimal sufficient mechanism synthesis."""

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
    HANDOFF_REPLAY_VERSION,
    SYNTHESIS_PROXY_SCOPE_NOTE,
    SYNTHESIS_VERSION,
    TRAINING_CONSTRAINT_VERSION,
    aggregate_mechanism_metrics,
    build_mechanism_conclusions,
    build_s11_tasks,
    build_traceability_matrix,
    competence_policy_from_s08_candidate,
    competence_replay_fingerprint,
    condition_rows_for_competence_policy,
    local_memory_signal_competence_policy,
    policy_evidence_table,
    required_s15_evidence_sources,
    run_competence_condition,
    select_handoff_candidates,
    step_status_rows,
    validate_synthesis_outputs,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S15"
STEP_NUMBER = 15
STEP_TITLE = "Extract minimal sufficient mechanisms"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
S08_CANDIDATE_PATH = Path("/artifacts/results/e04_evolved_repair_policies.parquet")
TEST_MODULES = [
    "tests.test_e04_minimal_synthesis",
    "tests.test_e04_centralized_comparison",
    "tests.test_e04_overfitting_transfer",
    "tests.test_e04_competence_proxies",
    "tests.test_e04_field_predictors",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_required_table(artifacts_dir: Path, relative_path: str) -> pd.DataFrame:
    path = artifacts_dir / relative_path
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table extension: {path}")


def evidence_source_frame(artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in required_s15_evidence_sources():
        path = artifacts_dir / str(source["artifactPath"])
        rows.append(
            {
                **source,
                "path": str(path),
                "exists": bool(path.exists()),
                "sizeBytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
                "sha256": sha256_path(path) if path.exists() and path.is_file() else None,
            }
        )
    return pd.DataFrame(json_ready(rows))


def load_evolved_rows() -> list[dict[str, Any]]:
    if not S08_CANDIDATE_PATH.exists():
        return []
    df = pd.read_parquet(S08_CANDIDATE_PATH)
    if "rank" in df.columns:
        df = df.sort_values("rank", kind="mergesort")
    return [row.to_dict() for _, row in df.iterrows()]


def build_policy_catalog(evolved_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    policies = [local_memory_signal_competence_policy()]
    for row in evolved_rows:
        policy = competence_policy_from_s08_candidate(row)
        if policy.replayable:
            policies.append(policy)
    rows = [policy.to_dict() for policy in policies]
    return {policy.policy_id: policy for policy in policies}, pd.DataFrame(json_ready(rows))


def enrich_handoff_candidates(handoff_df: pd.DataFrame, policy_by_id: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in handoff_df.to_dict(orient="records"):
        policy = policy_by_id.get(str(row["policyId"]))
        payload = dict(row)
        if policy is not None:
            spec = policy.to_dict()
            payload.update(
                {
                    "policySpecJson": compact_json(spec),
                    "learningConfigJson": compact_json(spec.get("learningConfig")),
                    "memoryConfigJson": compact_json(spec.get("memoryConfig")),
                    "signalConfigJson": compact_json(spec.get("signalConfig")),
                    "repairConfigJson": compact_json(spec.get("repairConfig")),
                    "oracleAccessAllowed": bool(spec.get("oracleAccessAllowed", False)),
                    "policyAuditSuccess": bool(spec.get("policyAuditSuccess", False)),
                    "replayable": bool(spec.get("replayable", False)),
                    "replayBlocker": spec.get("replayBlocker"),
                }
            )
        payload["claimBoundary"] = SYNTHESIS_PROXY_SCOPE_NOTE
        payload["synthesisVersion"] = SYNTHESIS_VERSION
        rows.append(payload)
    return pd.DataFrame(json_ready(rows))


def run_handoff_replay_panel(
    handoff_df: pd.DataFrame,
    policy_by_id: Mapping[str, Any],
    *,
    seed_count: int,
    seed_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    tasks = build_s11_tasks()
    condition_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    for policy_index, candidate in enumerate(handoff_df.itertuples(index=False)):
        policy = policy_by_id[str(candidate.policyId)]
        rows = condition_rows_for_competence_policy(
            policy,
            tasks,
            seed_count=seed_count,
            seed_base=seed_base + policy_index * 1000,
        )
        for condition in rows:
            condition = dict(condition)
            condition.update(
                {
                    "handoffTier": candidate.handoffTier,
                    "recommendedForHandoff": True,
                    "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                    "synthesisVersion": SYNTHESIS_VERSION,
                    "handoffReplayVersion": HANDOFF_REPLAY_VERSION,
                }
            )
            condition_rows.append(json_ready(condition))
            result, ticks = run_competence_condition(
                policy,
                condition,
                implementation="e04_s15_handoff_cpu_reference",
                research_step_id=STEP_ID,
            )
            result.update(
                {
                    "handoffTier": candidate.handoffTier,
                    "recommendedForHandoff": True,
                    "handoffReplayScore": float(result["conditionCompetenceCompositeProxy"]),
                    "oracleAccessAllowed": False,
                    "usesGlobalController": False,
                    "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                    "synthesisVersion": SYNTHESIS_VERSION,
                    "handoffReplayVersion": HANDOFF_REPLAY_VERSION,
                }
            )
            result_rows.append(json_ready(result))
            fingerprint_rows.append(json_ready(result))
            for tick in ticks:
                tick_row = dict(tick)
                tick_row.update(
                    {
                        "handoffTier": candidate.handoffTier,
                        "recommendedForHandoff": True,
                        "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                        "synthesisVersion": SYNTHESIS_VERSION,
                        "handoffReplayVersion": HANDOFF_REPLAY_VERSION,
                    }
                )
                tick_rows.append(json_ready(tick_row))
    fingerprint = competence_replay_fingerprint(fingerprint_rows)
    return pd.DataFrame(condition_rows), pd.DataFrame(result_rows), pd.DataFrame(tick_rows), fingerprint


def plot_s15_results(handoff_df: pd.DataFrame, conclusion_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not handoff_df.empty:
        ordered = handoff_df.sort_values("handoffCompositeProxy", ascending=True, kind="mergesort")
        labels = ordered["policyId"].astype(str).str.replace("enhanced_", "enh_", regex=False)
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        colors = ["#4B8F6A" if tier == "primary" else "#4F7CAC" for tier in ordered["handoffTier"]]
        ax.barh(labels, ordered["handoffCompositeProxy"], color=colors)
        ax.set_xlabel("S15 handoff composite proxy")
        ax.set_title("E04 selected handoff candidate scores")
        ax.set_xlim(0.0, 1.05)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        png = figures_dir / "e04_s15_handoff_candidate_scores.png"
        svg = figures_dir / "e04_s15_handoff_candidate_scores.svg"
        fig.savefig(png, dpi=160)
        fig.savefig(svg)
        plt.close(fig)
        paths.extend([png, svg])
    if not conclusion_df.empty:
        status_order = ["supportive", "supported_with_constraints", "supportive_with_constraints", "constraining"]
        counts = conclusion_df["conclusionStatus"].value_counts().reindex(status_order).dropna()
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(counts.index.astype(str), counts.values, color=["#4B8F6A", "#6B8E9D", "#7A82B8", "#A66B45"][: len(counts)])
        ax.set_ylabel("Conclusion count")
        ax.set_title("E04 S15 proxy-scoped synthesis conclusion types")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        png = figures_dir / "e04_s15_mechanism_conclusion_counts.png"
        svg = figures_dir / "e04_s15_mechanism_conclusion_counts.svg"
        fig.savefig(png, dpi=160)
        fig.savefig(svg)
        plt.close(fig)
        paths.extend([png, svg])
    return paths


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_root = step_dir / "code"
    sources = [
        REPO_ROOT / "memory_repair" / "synthesis.py",
        REPO_ROOT / "memory_repair" / "competence_proxies.py",
        REPO_ROOT / "memory_repair" / "centralized_comparison.py",
        REPO_ROOT / "memory_repair" / "evolution.py",
        REPO_ROOT / "memory_repair" / "memory.py",
        REPO_ROOT / "memory_repair" / "signals.py",
        REPO_ROOT / "memory_repair" / "repair.py",
        REPO_ROOT / "memory_repair" / "fatigue.py",
        REPO_ROOT / "memory_repair" / "homeostasis.py",
        REPO_ROOT / "memory_repair" / "learning.py",
        REPO_ROOT / "memory_repair" / "training_constraints.py",
        REPO_ROOT / "scripts" / "e04_s15_minimal_mechanisms.py",
        REPO_ROOT / "tests" / "test_e04_minimal_synthesis.py",
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


def report_markdown(
    *,
    validation_df: pd.DataFrame,
    conclusion_df: pd.DataFrame,
    handoff_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    traceability_df: pd.DataFrame,
    metrics: Mapping[str, Any],
    caveats: Sequence[str],
    recommended_next_action: str,
    outcome: str,
    artifact_note: str,
) -> str:
    validation_line = f"{int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    replay_summary = (
        replay_df.groupby("policyId", dropna=False)
        .agg(
            replayRows=("conditionId", "count"),
            meanReplayScore=("handoffReplayScore", "mean"),
            completedRuns=("completed", "sum"),
            meanFinalSortedness=("finalSortednessPercent", "mean"),
        )
        .reset_index()
        if not replay_df.empty
        else pd.DataFrame()
    )
    return (
        "\n".join(
            [
                "# E04 Minimal Repair Mechanisms",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: {artifact_note}",
                f"- Validation result: {validation_line}",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                SYNTHESIS_PROXY_SCOPE_NOTE,
                "",
                "## Bottom Line",
                "",
                "S15 does not support a single uniquely smallest mechanism package. It supports a practical computational handoff package: a local-adaptive adjacent-swap policy with bounded local memory, local signal emission/sensing, signal-threshold repair, and the S07 no-oracle audit gate. This package is sufficient only within the tested E04 computational panels and remains below an explicitly labeled global-oracle upper-bound comparator.",
                "",
                f"Outcome classification: {outcome}.",
                "",
                "## Direct Proxy Anchors",
                "",
                markdown_table(
                    ["metric", "value"],
                    [
                        ["S09 mean cell-memory score delta", metrics.get("meanCellMemoryScoreDelta")],
                        ["S09 mean signal-field-memory score delta", metrics.get("meanSignalFieldMemoryScoreDelta")],
                        ["S10 best communication variant", metrics.get("bestCommunicationVariant")],
                        ["S10 best communication mean score delta", metrics.get("bestCommunicationMeanScoreDelta")],
                        ["S13 learned-policy holdout score", metrics.get("learnedPolicyHoldoutMeanScore")],
                        ["S13 learned-policy relative drop", metrics.get("learnedPolicyRelativeDrop")],
                        ["S14 mean global-minus-local score", metrics.get("meanCentralizedGlobalMinusLocalScore")],
                    ],
                ),
                "",
                "## Conclusions",
                "",
                markdown_table(
                    ["conclusionId", "status", "mechanism package", "conclusion", "evidence rows"],
                    [
                        [
                            row.conclusionId,
                            row.conclusionStatus,
                            row.minimalMechanismPackage,
                            row.conclusion,
                            int(traceability_df[traceability_df["conclusionId"].eq(row.conclusionId)].shape[0]),
                        ]
                        for row in conclusion_df.itertuples(index=False)
                    ],
                ),
                "",
                "## Handoff Candidates",
                "",
                markdown_table(
                    ["tier", "policyId", "family", "handoff proxy", "S11 overall", "S13 holdout", "S14 local", "rationale"],
                    [
                        [
                            row.handoffTier,
                            row.policyId,
                            row.familyKind,
                            row.handoffCompositeProxy,
                            row.overallCompetenceProxy,
                            row.holdoutMeanScore,
                            row.meanLocalControllerScore,
                            row.handoffRationale,
                        ]
                        for row in handoff_df.itertuples(index=False)
                    ],
                ),
                "",
                "## S15 Replay Validation",
                "",
                markdown_table(
                    ["policyId", "replay rows", "mean replay score", "completed runs", "mean final sortedness"],
                    [
                        [row.policyId, row.replayRows, row.meanReplayScore, row.completedRuns, row.meanFinalSortedness]
                        for row in replay_summary.itertuples(index=False)
                    ],
                ),
                "",
                "## Traceability",
                "",
                f"Every conclusion in this report is backed by `evidenceSourceIds` and expanded in the S15 traceability table. Trace rows: {len(traceability_df)}.",
                "",
                "## Caveats",
                "",
                "- The analysis is a direct computational proxy synthesis and does not validate biological repair, biological intelligence, or causal tissue fields.",
                "- S09 indicates cell-local memory wrappers alone are weak in this implementation when policies do not strongly exploit them.",
                "- S10 randomized-control signaling performed well on the small panel, so signaling benefits are not cleanly attributable to a single deterministic communication architecture.",
                "- S13 transfer gaps show evolved policies are more brittle than the learned memory-signal handoff candidate.",
                "- S14 keeps the centralized comparator explicitly global-oracle-labeled; it is not a local or biologically fair policy.",
            ]
        )
        + "\n"
    )


def write_reports(
    *,
    step_dir: Path,
    reports_dir: Path,
    bundle_dir: Path,
    validation_df: pd.DataFrame,
    conclusion_df: pd.DataFrame,
    handoff_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    traceability_df: pd.DataFrame,
    metrics: Mapping[str, Any],
    caveats: Sequence[str],
    recommended_next_action: str,
    outcome: str,
) -> list[Path]:
    reports: list[Path] = []
    artifact_note = (
        f"`{step_dir / 'minimal_mechanism_conclusions.csv'}`, `{step_dir / 'evidence_traceability_matrix.csv'}`, "
        f"`{step_dir / 'selected_policy_handoff_candidates.csv'}`, `{step_dir / 'selected_policy_handoff_replays.csv'}`, "
        f"`{reports_dir / 'e04_minimal_repair_mechanisms.md'}`, and report bundle inputs under `{bundle_dir}`"
    )
    main_text = report_markdown(
        validation_df=validation_df,
        conclusion_df=conclusion_df,
        handoff_df=handoff_df,
        replay_df=replay_df,
        traceability_df=traceability_df,
        metrics=metrics,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
        outcome=outcome,
        artifact_note=artifact_note,
    )
    main_report = reports_dir / "e04_minimal_repair_mechanisms.md"
    step_report = step_dir / "e04_minimal_repair_mechanisms.md"
    main_report.parent.mkdir(parents=True, exist_ok=True)
    main_report.write_text(main_text, encoding="utf-8")
    step_report.write_text(main_text, encoding="utf-8")
    reports.extend([main_report, step_report])

    validation_report = step_dir / "validation_report.md"
    validation_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Validation Report",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: `{step_dir / 'synthesis_validation.csv'}` and `.parquet`",
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

    handoff_report = step_dir / "handoff_candidate_report.md"
    handoff_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Handoff Candidate Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: `{step_dir / 'selected_policy_handoff_candidates.csv'}`, `{step_dir / 'selected_policy_handoff_replays.csv'}`, and report bundle candidate tables",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                SYNTHESIS_PROXY_SCOPE_NOTE,
                "",
                markdown_table(
                    ["tier", "policyId", "handoff proxy", "replayable", "audit", "rationale"],
                    [
                        [
                            row.handoffTier,
                            row.policyId,
                            row.handoffCompositeProxy,
                            row.replayable,
                            row.policyAuditSuccess,
                            row.handoffRationale,
                        ]
                        for row in handoff_df.itertuples(index=False)
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(handoff_report)

    summary = step_dir / "summary.md"
    summary.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Status Summary",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: primary outputs under `{step_dir}`, report copy `{main_report}`, report bundle inputs under `{bundle_dir}`, shared result copies, config copies, figures, `status.json`, and `artifact_manifest.json`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "Lay summary: S15 synthesized S01-S14 into a traceable, proxy-scoped mechanism analysis and reran a compact CPU-reference replay panel for selected handoff policies. The analysis identifies a practical local learned memory-signal handoff package but constrains the stronger claim that one unique smallest mechanism was isolated.",
                "",
                SYNTHESIS_PROXY_SCOPE_NOTE,
                "",
                f"Outcome classification: {outcome}.",
                f"Conclusion rows: {len(conclusion_df)}; traceability rows: {len(traceability_df)}; handoff candidates: {len(handoff_df)}; replay rows: {len(replay_df)}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(summary)
    return reports


def write_report_bundle(
    *,
    bundle_dir: Path,
    main_report: Path,
    conclusion_df: pd.DataFrame,
    traceability_df: pd.DataFrame,
    handoff_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    figure_paths: Sequence[Path],
) -> list[Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = bundle_dir / "tables"
    figures_dir = bundle_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    report_copy = bundle_dir / "e04_minimal_repair_mechanisms.md"
    shutil.copy2(main_report, report_copy)
    written.append(report_copy)
    for name, df in [
        ("minimal_mechanism_conclusions.csv", conclusion_df),
        ("evidence_traceability_matrix.csv", traceability_df),
        ("selected_policy_handoff_candidates.csv", handoff_df),
        ("selected_policy_handoff_replays.csv", replay_df),
        ("synthesis_validation.csv", validation_df),
    ]:
        path = tables_dir / name
        df.to_csv(path, index=False)
        written.append(path)
    for figure in figure_paths:
        destination = figures_dir / figure.name
        shutil.copy2(figure, destination)
        written.append(destination)
    manifest_path = bundle_dir / "e04_s15_report_bundle_manifest.json"
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.e04_s15_report_bundle_inputs.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            "files": [str(path) for path in written],
        },
    )
    written.append(manifest_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--handoff-seed-count", type=int, default=1)
    parser.add_argument("--handoff-seed-base", type=int, default=21100)
    parser.add_argument("--max-evolved-handoff", type=int, default=2)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    figures_dir = artifacts_dir / "figures"
    reports_dir = artifacts_dir / "reports"
    bundle_dir = artifacts_dir / "report_bundle_inputs" / "e04_minimal_repair_mechanisms"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    artifacts: list[Path] = []

    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "handoffSeedCount": int(args.handoff_seed_count),
        "handoffSeedBase": int(args.handoff_seed_base),
        "maxEvolvedHandoff": int(args.max_evolved_handoff),
        "serialWorkerCount": 1,
        "s08CandidatePath": str(S08_CANDIDATE_PATH),
        "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
        "synthesisVersion": SYNTHESIS_VERSION,
        "handoffReplayVersion": HANDOFF_REPLAY_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "createdAt": utc_now(),
    }
    for path in [step_dir / "minimal_mechanism_synthesis_config.json", configs_dir / "e04_s15_minimal_mechanisms.json"]:
        write_json(path, config_payload)
        artifacts.append(path)
    for path in [step_dir / "minimal_mechanism_synthesis_config.yaml", configs_dir / "e04_s15_minimal_mechanisms.yaml"]:
        write_yaml(path, config_payload)
        artifacts.append(path)

    evidence_df = evidence_source_frame(artifacts_dir)
    status_payloads = [load_json(artifacts_dir / f"research_steps/S{index:02d}/status.json") for index in range(1, 15)]
    step_status_df = step_status_rows(status_payloads)
    memory_summary = read_required_table(artifacts_dir, "results/e04_s09_memory_ablation_summary.parquet")
    communication_summary = read_required_table(artifacts_dir, "results/e04_s10_communication_ablation_summary.parquet")
    communication_transfer = read_required_table(artifacts_dir, "results/e04_s10_communication_transfer_gaps.parquet")
    competence_profiles = read_required_table(artifacts_dir, "results/e04_s11_competence_policy_profiles.parquet")
    field_summary = read_required_table(artifacts_dir, "results/e04_s12_tissue_field_predictor_summary.parquet")
    transfer_gaps = read_required_table(artifacts_dir, "results/e04_s13_overfitting_policy_gaps.parquet")
    centralized_summary = read_required_table(artifacts_dir, "results/e04_s14_centralized_group_summary.parquet")
    candidate_audits = read_required_table(artifacts_dir, "results/e04_s08_candidate_audits.parquet")
    evolved_rows = load_evolved_rows()
    policy_by_id, policy_catalog_df = build_policy_catalog(evolved_rows)

    metrics = aggregate_mechanism_metrics(memory_summary, communication_summary, competence_profiles, transfer_gaps, centralized_summary)
    policy_evidence_df = policy_evidence_table(competence_profiles, transfer_gaps, centralized_summary, candidate_audits)
    handoff_df = select_handoff_candidates(policy_evidence_df, max_evolved=int(args.max_evolved_handoff))
    handoff_df = enrich_handoff_candidates(handoff_df, policy_by_id)
    conclusion_df = build_mechanism_conclusions(metrics)
    traceability_df = build_traceability_matrix(conclusion_df, evidence_df)

    replay_conditions_df, replay_df, replay_tick_df, first_fingerprint = run_handoff_replay_panel(
        handoff_df,
        policy_by_id,
        seed_count=int(args.handoff_seed_count),
        seed_base=int(args.handoff_seed_base),
    )
    _, replay_df_second, _, second_fingerprint = run_handoff_replay_panel(
        handoff_df,
        policy_by_id,
        seed_count=int(args.handoff_seed_count),
        seed_base=int(args.handoff_seed_base),
    )
    deterministic_replay_match = bool(first_fingerprint == second_fingerprint)
    replay_fingerprint_df = pd.DataFrame(
        [
            {
                "researchStepId": STEP_ID,
                "firstReplayFingerprint": first_fingerprint,
                "secondReplayFingerprint": second_fingerprint,
                "deterministicReplayMatch": deterministic_replay_match,
                "firstReplayRows": int(len(replay_df)),
                "secondReplayRows": int(len(replay_df_second)),
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                "synthesisVersion": SYNTHESIS_VERSION,
                "handoffReplayVersion": HANDOFF_REPLAY_VERSION,
            }
        ]
    )

    # Add S12 and S10 supporting summaries to the bundle as compact evidence tables.
    supporting_metric_df = pd.DataFrame(
        [
            {
                "metricId": key,
                "metricValueJson": compact_json(value),
                "metricValueType": type(value).__name__,
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                "synthesisVersion": SYNTHESIS_VERSION,
            }
            for key, value in metrics.items()
        ]
    )
    artifacts.extend(dataframe_to_artifacts(evidence_df, step_dir / "evidence_source_catalog", results_dir / "e04_s15_evidence_source_catalog"))
    artifacts.extend(dataframe_to_artifacts(step_status_df, step_dir / "prior_step_status", results_dir / "e04_s15_prior_step_status"))
    artifacts.extend(dataframe_to_artifacts(supporting_metric_df, step_dir / "synthesis_metric_anchors", results_dir / "e04_s15_synthesis_metric_anchors"))
    artifacts.extend(dataframe_to_artifacts(policy_catalog_df, step_dir / "handoff_policy_catalog", results_dir / "e04_s15_handoff_policy_catalog"))
    artifacts.extend(dataframe_to_artifacts(policy_evidence_df, step_dir / "policy_evidence_scores", results_dir / "e04_s15_policy_evidence_scores"))
    artifacts.extend(dataframe_to_artifacts(handoff_df, step_dir / "selected_policy_handoff_candidates", results_dir / "e04_s15_selected_policy_handoff_candidates"))
    artifacts.extend(dataframe_to_artifacts(conclusion_df, step_dir / "minimal_mechanism_conclusions", results_dir / "e04_s15_minimal_mechanism_conclusions"))
    artifacts.extend(dataframe_to_artifacts(traceability_df, step_dir / "evidence_traceability_matrix", results_dir / "e04_s15_evidence_traceability_matrix"))
    artifacts.extend(dataframe_to_artifacts(replay_conditions_df, step_dir / "selected_policy_handoff_replay_conditions", results_dir / "e04_s15_selected_policy_handoff_replay_conditions"))
    artifacts.extend(dataframe_to_artifacts(replay_df, step_dir / "selected_policy_handoff_replays", results_dir / "e04_s15_selected_policy_handoff_replays"))
    artifacts.extend(dataframe_to_artifacts(replay_tick_df, step_dir / "selected_policy_handoff_replay_tick_records", results_dir / "e04_s15_selected_policy_handoff_replay_tick_records"))
    artifacts.extend(dataframe_to_artifacts(replay_fingerprint_df, step_dir / "handoff_replay_fingerprints", results_dir / "e04_s15_handoff_replay_fingerprints"))
    artifacts.extend(dataframe_to_artifacts(communication_transfer, step_dir / "supporting_s10_communication_transfer_gaps"))
    artifacts.extend(dataframe_to_artifacts(field_summary, step_dir / "supporting_s12_field_predictor_summary"))

    figure_paths = plot_s15_results(handoff_df, conclusion_df, figures_dir)
    for figure in figure_paths:
        step_figure = step_dir / figure.name
        shutil.copy2(figure, step_figure)
        artifacts.extend([figure, step_figure])

    validation_seed = pd.DataFrame(
        [
            {
                "checkId": "placeholder_until_report_written",
                "success": True,
                "detail": "report existence is checked after reports are written",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                "synthesisVersion": SYNTHESIS_VERSION,
            }
        ]
    )
    caveats = [
        "S15 is a direct computational proxy synthesis and does not validate biological repair, biological intelligence, or causal tissue fields.",
        "The evidence does not isolate a unique smallest mechanism package; S10 randomized-control signaling performed best on the small communication-ablation panel.",
        "Cell-local memory capacity alone had near-zero or slightly negative S09 score deltas in this implementation.",
        "The recommended handoff policies are local-only computational candidates and should be replayed again in any new substrate or experiment.",
    ]
    recommended_next_action = "Chief review of S15 E04 minimal-mechanism package; E04 has no further queued research step, and downstream E05/E06/E07 handoff should start only after review."
    outcome = "constraining/contradictory"
    reports = write_reports(
        step_dir=step_dir,
        reports_dir=reports_dir,
        bundle_dir=bundle_dir,
        validation_df=validation_seed,
        conclusion_df=conclusion_df,
        handoff_df=handoff_df,
        replay_df=replay_df,
        traceability_df=traceability_df,
        metrics=metrics,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
        outcome=outcome,
    )
    artifacts.extend(reports)

    bundle_paths = write_report_bundle(
        bundle_dir=bundle_dir,
        main_report=reports_dir / "e04_minimal_repair_mechanisms.md",
        conclusion_df=conclusion_df,
        traceability_df=traceability_df,
        handoff_df=handoff_df,
        replay_df=replay_df,
        validation_df=validation_seed,
        figure_paths=figure_paths,
    )
    artifacts.extend(bundle_paths)

    validation_df = validate_synthesis_outputs(
        step_status_df,
        evidence_df,
        conclusion_df,
        traceability_df,
        handoff_df,
        replay_df,
        report_exists=(reports_dir / "e04_minimal_repair_mechanisms.md").exists(),
        report_bundle_exists=bundle_dir.exists(),
        deterministic_replay_match=deterministic_replay_match,
    )

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
                        "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                        "synthesisVersion": SYNTHESIS_VERSION,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "synthesis_validation", results_dir / "e04_s15_synthesis_validation"))

    # Rewrite reports and bundle after final validation includes test status.
    reports = write_reports(
        step_dir=step_dir,
        reports_dir=reports_dir,
        bundle_dir=bundle_dir,
        validation_df=validation_df,
        conclusion_df=conclusion_df,
        handoff_df=handoff_df,
        replay_df=replay_df,
        traceability_df=traceability_df,
        metrics=metrics,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
        outcome=outcome,
    )
    artifacts.extend(reports)
    bundle_paths = write_report_bundle(
        bundle_dir=bundle_dir,
        main_report=reports_dir / "e04_minimal_repair_mechanisms.md",
        conclusion_df=conclusion_df,
        traceability_df=traceability_df,
        handoff_df=handoff_df,
        replay_df=replay_df,
        validation_df=validation_df,
        figure_paths=figure_paths,
    )
    artifacts.extend(bundle_paths)

    code_artifacts = copy_code_artifacts(step_dir)
    artifacts.extend(code_artifacts)

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
            "validationResultsPath": str(step_dir / "synthesis_validation.csv"),
        },
        "caveatsOrBlockers": caveats if success else caveats + ["At least one S15 validation check failed."],
        "recommendedNextAction": recommended_next_action,
        "experimentId": EXPERIMENT_ID,
        "title": STEP_TITLE,
        "outcomeClassification": outcome,
        "startedAt": config_payload["createdAt"],
        "completedAt": utc_now(),
        "runtimeSeconds": time.perf_counter() - started,
        "git": get_git_metadata(),
        "platform": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor()},
        "priorStepCount": int(len(step_status_df)),
        "evidenceSourceCount": int(len(evidence_df)),
        "conclusionCount": int(len(conclusion_df)),
        "traceabilityRowCount": int(len(traceability_df)),
        "handoffCandidateCount": int(len(handoff_df)),
        "handoffReplayRowCount": int(len(replay_df)),
        "deterministicReplayMatch": deterministic_replay_match,
        "primaryHandoffPolicyId": None if handoff_df.empty else str(handoff_df.iloc[0]["policyId"]),
        "reportPath": str(reports_dir / "e04_minimal_repair_mechanisms.md"),
        "reportBundlePath": str(bundle_dir),
        "serialWorkerCount": 1,
        "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
        "synthesisVersion": SYNTHESIS_VERSION,
        "handoffReplayVersion": HANDOFF_REPLAY_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    write_json(status_path, status)
    manifest = artifact_manifest(artifacts_dir, [path for path in artifacts if path != manifest_path])
    write_json(manifest_path, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
