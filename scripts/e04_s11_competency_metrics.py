#!/usr/bin/env python3
"""Run E04 S11 intelligence-like competency benchmarks."""

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

from src.e04.competency_metrics import (
    REQUIRED_S11_POLICY_MODES,
    S11_COMPETENCY_AXES,
    S11_DEGRADATION_LEVELS,
    S11_PROTOCOL_ID,
    assert_s11_design,
    default_s11_eval_configs,
    default_s11_policy_modes,
    run_s11_competency_matrix,
)
from src.e04.memory_ablations import load_selected_s08_policies
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
)


STEP_ID = "S11"
STEP_NUMBER = 11
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument(
        "--selected-policies",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "policies" / "e04_evolved_repair_policies.jsonl",
    )
    parser.add_argument(
        "--s09-delta-summary",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e04_memory_ablations_delta_summary.csv",
    )
    parser.add_argument(
        "--s10-delta-summary",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e04_communication_ablations_delta_summary.csv",
    )
    parser.add_argument("--max-events", type=int, default=180)
    parser.add_argument("--transfer-max-events", type=int, default=220)
    parser.add_argument("--policy-limit", type=int, default=None)
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


def _parse_json(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str) and value:
        return json.loads(value)
    if isinstance(value, Mapping):
        return value
    return {}


def load_prior_findings(args: argparse.Namespace) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "s09DeltaSummaryPath": str(args.s09_delta_summary),
        "s10DeltaSummaryPath": str(args.s10_delta_summary),
        "s09NeighborMemorySupported": False,
        "s10CommunicationReliabilityMode": "unknown",
    }
    if args.s09_delta_summary.exists():
        s09 = pd.read_csv(args.s09_delta_summary)
        neighbor = s09[s09["memory_ablation"] == "neighbor_memory"]
        if not neighbor.empty:
            row = neighbor.iloc[0].to_dict()
            finding.update(
                {
                    "s09NeighborMemorySupported": bool(
                        float(row["mean_fitness_delta_vs_no_memory"]) > 0.0
                        and float(row["positive_fitness_pair_fraction"]) >= 0.60
                    ),
                    "s09NeighborMemoryMeanDelta": float(row["mean_fitness_delta_vs_no_memory"]),
                    "s09NeighborMemoryPositivePairFraction": float(row["positive_fitness_pair_fraction"]),
                }
            )
    if args.s10_delta_summary.exists():
        s10 = pd.read_csv(args.s10_delta_summary)
        passing = s10[
            (s10["mean_fitness_delta_vs_memory_only"] > 0.0)
            & (s10["positive_fitness_pair_fraction"] >= 0.60)
        ]
        finding.update(
            {
                "s10CommunicationReliabilityMode": "none" if passing.empty else str(passing.iloc[0]["communication_ablation"]),
                "s10Rows": int(len(s10)),
            }
        )
    return finding


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["policy_mode_order", "policy_mode", "policy_mode_label", "competency_axis"], as_index=False)
        .agg(
            runs=("competency_score", "size"),
            policies=("s08_source_policy_id", "nunique"),
            mean_competency_score=("competency_score", "mean"),
            mean_axis_score=("competency_axis_score", "mean"),
            mean_fitness=("fitness_score", "mean"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_recovery_rate=("recovery_rate", "mean"),
            mean_route_diversity_score=("route_diversity_score", "mean"),
            mean_energy_total=("energy_total", "mean"),
            oracle_hit_rows=("uses_global_oracle", "sum"),
        )
        .sort_values(["policy_mode_order", "competency_axis"])
        .reset_index(drop=True)
    )
    for column in [
        "mean_competency_score",
        "mean_axis_score",
        "mean_fitness",
        "mean_time_in_target_fraction",
        "mean_final_sortedness_percent",
        "mean_recovery_rate",
        "mean_route_diversity_score",
        "mean_energy_total",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def matched_deltas(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "s08_source_policy_id",
        "competency_axis",
        "scenario_name",
        "damage_level",
        "activation_seed",
        "policy_seed",
        "schedule_seed",
        "array_size",
        "max_events",
    ]
    metrics = [
        "competency_score",
        "competency_axis_score",
        "fitness_score",
        "time_in_target_fraction",
        "final_sortedness_percent",
        "recovery_rate",
        "route_diversity_score",
        "energy_total",
        "schedule_sha256",
    ]
    open_loop = df[df["policy_mode"] == "original_open_loop_bubble"][[*key_cols, *metrics]].rename(
        columns={column: f"open_loop_{column}" for column in metrics}
    )
    neighbor = df[df["policy_mode"] == "neighbor_memory"][[*key_cols, *metrics]].rename(
        columns={column: f"neighbor_memory_{column}" for column in metrics}
    )
    merged = df.merge(open_loop, on=key_cols, how="left", validate="many_to_one")
    merged = merged.merge(neighbor, on=key_cols, how="left", validate="many_to_one")
    for metric in metrics:
        if metric == "schedule_sha256":
            continue
        merged[f"{metric}_delta_vs_open_loop"] = merged[metric] - merged[f"open_loop_{metric}"]
        merged[f"{metric}_delta_vs_neighbor_memory"] = merged[metric] - merged[f"neighbor_memory_{metric}"]
    merged["matched_open_loop_schedule"] = merged["schedule_sha256"] == merged["open_loop_schedule_sha256"]
    merged["matched_neighbor_memory_schedule"] = merged["schedule_sha256"] == merged["neighbor_memory_schedule_sha256"]
    return merged[merged["policy_mode"] != "original_open_loop_bubble"].copy()


def summarize_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    summary = (
        delta_df.groupby(["policy_mode_order", "policy_mode", "policy_mode_label"], as_index=False)
        .agg(
            matched_rows=("competency_score_delta_vs_open_loop", "size"),
            mean_competency_delta_vs_open_loop=("competency_score_delta_vs_open_loop", "mean"),
            positive_competency_fraction_vs_open_loop=(
                "competency_score_delta_vs_open_loop",
                lambda values: float((values > 0).mean()),
            ),
            mean_axis_delta_vs_open_loop=("competency_axis_score_delta_vs_open_loop", "mean"),
            mean_fitness_delta_vs_open_loop=("fitness_score_delta_vs_open_loop", "mean"),
            mean_competency_delta_vs_neighbor_memory=("competency_score_delta_vs_neighbor_memory", "mean"),
            positive_competency_fraction_vs_neighbor_memory=(
                "competency_score_delta_vs_neighbor_memory",
                lambda values: float((values > 0).mean()),
            ),
            matched_open_loop_schedule_rows=("matched_open_loop_schedule", "sum"),
            matched_neighbor_memory_schedule_rows=("matched_neighbor_memory_schedule", "sum"),
        )
        .sort_values("policy_mode_order")
        .reset_index(drop=True)
    )
    for column in [
        "mean_competency_delta_vs_open_loop",
        "positive_competency_fraction_vs_open_loop",
        "mean_axis_delta_vs_open_loop",
        "mean_fitness_delta_vs_open_loop",
        "mean_competency_delta_vs_neighbor_memory",
        "positive_competency_fraction_vs_neighbor_memory",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def degradation_profile(df: pd.DataFrame) -> pd.DataFrame:
    graceful = df[df["competency_axis"] == "graceful_degradation"].copy()
    if graceful.empty:
        return pd.DataFrame()
    profile = (
        graceful.groupby(["policy_mode_order", "policy_mode", "policy_mode_label", "damage_level"], as_index=False)
        .agg(
            runs=("competency_score", "size"),
            mean_competency_score=("competency_score", "mean"),
            mean_fitness=("fitness_score", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_recovery_rate=("recovery_rate", "mean"),
        )
        .sort_values(["policy_mode_order", "damage_level"])
        .reset_index(drop=True)
    )
    for column in ["mean_competency_score", "mean_fitness", "mean_final_sortedness_percent", "mean_recovery_rate"]:
        profile[column] = profile[column].round(6)
    slopes = []
    for mode, items in profile.groupby("policy_mode"):
        xs = items["damage_level"].astype(float)
        ys = items["mean_competency_score"].astype(float)
        if len(items) > 1:
            slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
        else:
            slope = 0.0
        slopes.append({"policy_mode": mode, "degradation_slope": slope})
    slope_df = pd.DataFrame(slopes)
    return profile.merge(slope_df, on="policy_mode", how="left")


def infer_competency_outcome(delta_summary_df: pd.DataFrame) -> dict[str, Any]:
    if delta_summary_df.empty:
        return {
            "outcome": "null",
            "reason": "no matched competency deltas were available",
            "criterion": "neighbor_memory mean competency delta vs open-loop > 0 and positive-row fraction >= 0.60",
        }
    neighbor = delta_summary_df[delta_summary_df["policy_mode"] == "neighbor_memory"]
    if neighbor.empty:
        return {
            "outcome": "null",
            "reason": "neighbor-memory rows were unavailable",
            "criterion": "neighbor_memory mean competency delta vs open-loop > 0 and positive-row fraction >= 0.60",
        }
    row = neighbor.iloc[0].to_dict()
    supportive = (
        float(row["mean_competency_delta_vs_open_loop"]) > 0.0
        and float(row["positive_competency_fraction_vs_open_loop"]) >= 0.60
    )
    return {
        "outcome": "supportive" if supportive else "constraining/contradictory",
        "criterion": "neighbor_memory mean competency delta vs open-loop > 0 and positive-row fraction >= 0.60",
        "neighborMeanCompetencyDeltaVsOpenLoop": float(row["mean_competency_delta_vs_open_loop"]),
        "neighborPositiveFractionVsOpenLoop": float(row["positive_competency_fraction_vs_open_loop"]),
    }


def write_figure(summary_df: pd.DataFrame, degradation_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = summary_df.pivot_table(
        index="policy_mode_label",
        columns="competency_axis",
        values="mean_competency_score",
        aggfunc="mean",
    )
    ordered_labels = (
        summary_df[["policy_mode_order", "policy_mode_label"]]
        .drop_duplicates()
        .sort_values("policy_mode_order")["policy_mode_label"]
        .tolist()
    )
    pivot = pivot.reindex(index=ordered_labels, columns=list(S11_COMPETENCY_AXES))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    image = axes[0].imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Competency score by held-out axis")
    axes[0].set_xticks(range(len(pivot.columns)), [item.replace("_", " ") for item in pivot.columns], rotation=35, ha="right")
    axes[0].set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    if not degradation_df.empty:
        for label, items in degradation_df.groupby("policy_mode_label", sort=False):
            ordered = items.sort_values("damage_level")
            axes[1].plot(
                ordered["damage_level"],
                ordered["mean_competency_score"],
                marker="o",
                linewidth=1.8,
                label=str(label),
            )
    axes[1].set_title("Graceful-degradation profile")
    axes[1].set_xlabel("Scheduled freeze damage level")
    axes[1].set_ylabel("Mean competency score")
    axes[1].set_xticks(list(S11_DEGRADATION_LEVELS))
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    *,
    df: pd.DataFrame,
    traces_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    configs: Sequence[Any],
    modes: Sequence[Any],
    selected_policy_count: int,
    design_audit: Mapping[str, Any],
    prior_findings: Mapping[str, Any],
    unit_tests: Mapping[str, Any],
    e04_s11_tests: Mapping[str, Any],
    e04_s10_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_rows = int(selected_policy_count * len(configs) * len(modes))
    rows.append(
        {
            "validation_case": "competency_matrix_complete",
            "success": bool(len(df) == expected_rows and tuple(sorted(df["policy_mode"].unique())) == tuple(sorted(REQUIRED_S11_POLICY_MODES))),
            "detail": f"rows={len(df)}; expected={expected_rows}; modes={sorted(df['policy_mode'].unique())}",
        }
    )
    group_cols = [
        "s08_source_policy_id",
        "competency_axis",
        "scenario_name",
        "damage_level",
        "activation_seed",
        "policy_seed",
        "schedule_seed",
        "array_size",
        "max_events",
    ]
    grouped = df.groupby(group_cols)
    complete_groups = grouped["policy_mode"].nunique().eq(len(REQUIRED_S11_POLICY_MODES)).all()
    matched_schedule_groups = grouped["schedule_sha256"].nunique().eq(1).all()
    rows.append(
        {
            "validation_case": "matched_seed_and_schedule_groups_complete",
            "success": bool(
                design_audit.get("success")
                and complete_groups
                and matched_schedule_groups
                and delta_df["matched_open_loop_schedule"].all()
                and delta_df["matched_neighbor_memory_schedule"].all()
            ),
            "detail": json.dumps(
                {
                    "designAudit": dict(design_audit),
                    "groupCount": int(grouped.ngroups),
                    "completeGroups": bool(complete_groups),
                    "matchedScheduleGroups": bool(matched_schedule_groups),
                },
                sort_keys=True,
                default=str,
            )[:1800],
        }
    )
    local_rows = df[df["mode_kind"] != "open_loop"]
    feature_records = [_parse_json(value) for value in local_rows["feature_audit_summary_json"]]
    signal_records = [_parse_json(value) for value in local_rows["signal_observed_json"]]
    forbidden_hits = [
        record
        for record in feature_records
        if record.get("forbiddenTokenHits") or record.get("excludedSignalHits") or record.get("listLikeFeatureValueHits")
    ]
    target_nonzero = [record for record in signal_records if not record.get("targetDerivedSignalFieldsZeroed", False)]
    rows.append(
        {
            "validation_case": "s07_projected_local_only_policy_inputs",
            "success": bool(
                (local_rows["uses_global_oracle"] == False).all()
                and all(not record.get("usesGlobalOracle", True) for record in feature_records)
                and local_rows["policy_input_contract"].str.startswith("S07 LocalTrainingObservation").all()
            ),
            "detail": f"local rows={len(local_rows)}; oracle rows={int(local_rows['uses_global_oracle'].sum())}; protocol={LOCAL_ONLY_PROTOCOL_ID}",
        }
    )
    rows.append(
        {
            "validation_case": "target_derived_and_forbidden_fields_excluded",
            "success": bool(not forbidden_hits and not target_nonzero),
            "detail": (
                f"feature forbidden/excluded hits={len(forbidden_hits)}; target-derived nonzero signal rows={len(target_nonzero)}; "
                f"allowed={list(ALLOWED_TRAINING_SIGNAL_FIELDS)}; excluded={list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}"
            ),
        }
    )
    run_ids = set(df["run_id"])
    trace_ids = set(traces_df["run_id"]) if not traces_df.empty else set()
    rows.append(
        {
            "validation_case": "metrics_computed_offline_from_traces",
            "success": bool(
                run_ids == trace_ids
                and (df["competency_metrics_offline_from_trace"] == True).all()
                and traces_df.groupby("run_id")["event_step"].min().eq(0).all()
            ),
            "detail": f"result run ids={len(run_ids)}; trace run ids={len(trace_ids)}; trace rows={len(traces_df)}",
        }
    )
    rows.append(
        {
            "validation_case": "heldout_axes_transfer_recovery_and_degradation_present",
            "success": bool(
                set(df["competency_axis"]) == set(S11_COMPETENCY_AXES)
                and set(df[df["competency_axis"] == "graceful_degradation"]["damage_level"]) == set(S11_DEGRADATION_LEVELS)
                and 12 in set(df["array_size"])
            ),
            "detail": f"axes={sorted(df['competency_axis'].unique())}; array sizes={sorted(df['array_size'].unique())}",
        }
    )
    rows.append(
        {
            "validation_case": "s09_s10_supported_mechanisms_and_controls_present",
            "success": bool(
                prior_findings.get("s09NeighborMemorySupported", False)
                and prior_findings.get("s10CommunicationReliabilityMode") == "none"
                and "neighbor_memory" in set(df["policy_mode"])
                and "s08_discovered_no_memory" in set(df["policy_mode"])
                and "original_open_loop_bubble" in set(df["policy_mode"])
            ),
            "detail": json.dumps(dict(prior_findings), sort_keys=True, default=str),
        }
    )
    rows.append(
        {
            "validation_case": "no_centralized_baseline_mixed_into_local_claims",
            "success": bool((df["centralized_baseline"] == False).all() and (df["eligible_for_local_only_claims"] == True).all()),
            "detail": f"centralized rows={int(df['centralized_baseline'].sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "unit_and_regression_tests_passed",
            "success": bool(
                unit_tests["success"]
                and e04_s11_tests["success"]
                and e04_s10_tests["success"]
                and e03_policy_tests["success"]
                and e02_tests["success"]
            ),
            "detail": (
                f"E04 discover={unit_tests['returnCode']}; E04 S11={e04_s11_tests['returnCode']}; "
                f"E04 S10={e04_s10_tests['returnCode']}; E03 policy={e03_policy_tests['returnCode']}; "
                f"E02 deterministic={e02_tests['returnCode']}"
            ),
        }
    )
    return pd.DataFrame(rows)


def render_report(
    *,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    delta_summary_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    outcome: Mapping[str, Any],
    prior_findings: Mapping[str, Any],
    validation_line: str,
    validation_success: bool,
    artifact_paths: Sequence[Path],
    result_path: Path,
    trace_path: Path,
    figure_path: Path,
    config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    args: argparse.Namespace,
    selected_policy_count: int,
    unit_tests: Mapping[str, Any],
    e04_s11_tests: Mapping[str, Any],
    e04_s10_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> str:
    classification = "constraining/contradictory" if not validation_success else str(outcome.get("outcome", "null"))
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{e04_s11_tests['command']}` -> return code {e04_s11_tests['returnCode']}",
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e04_s10_tests['command']}` -> return code {e04_s10_tests['returnCode']}",
            f"- `{e03_policy_tests['command']}` -> return code {e03_policy_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            (
                "- `python scripts/e04_s11_competency_metrics.py --repo-dir /workspace/cell-research "
                "--artifacts-dir $ARTIFACTS_DIR`"
            ),
        ]
    )
    summary_table = markdown_table(
        summary_df,
        [
            "policy_mode_label",
            "competency_axis",
            "runs",
            "mean_competency_score",
            "mean_axis_score",
            "mean_time_in_target_fraction",
            "mean_final_sortedness_percent",
            "mean_recovery_rate",
            "mean_route_diversity_score",
            "oracle_hit_rows",
        ],
        max_rows=40,
    )
    delta_table = markdown_table(
        delta_summary_df,
        [
            "policy_mode_label",
            "matched_rows",
            "mean_competency_delta_vs_open_loop",
            "positive_competency_fraction_vs_open_loop",
            "mean_competency_delta_vs_neighbor_memory",
            "positive_competency_fraction_vs_neighbor_memory",
        ],
        max_rows=20,
    )
    degradation_table = markdown_table(
        degradation_df,
        [
            "policy_mode_label",
            "damage_level",
            "mean_competency_score",
            "mean_fitness",
            "mean_final_sortedness_percent",
            "degradation_slope",
        ],
        max_rows=40,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)
    return f"""# E04 S11 Research Step Full Results

## Top Summary

- Research step ID: S11
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {classification}
- Caveats or blockers: "Intelligence-like" remains a computational proxy. The alternate-route/barrier metric is necessarily limited in a 1D adjacent-swap substrate, so route diversity is a behavioral proxy rather than true spatial path planning. Long-range scalar signaling remains less local and is reported as a sensitivity arm, not a supported local mechanism.
- Lay summary: S11 tested whether the S09-supported neighbor-memory mechanism and S10 signaling variants behave more competently than original/open-loop and no-memory discovered controls on held-out shocks, larger arrays, middle barriers, recovery shocks, and damage ladders. Policy choices for learned/local policies used only the S07 projected local view; scores were computed afterward from traces.
- Recommended next action: Hand control back to the Chief Scientist. If accepted, proceed to S12 emergent tissue-field analysis using the S11 traces and outcomes.

## Frozen Question

Do memory and signaling policies show goal attainment by multiple routes, barrier circumvention, recovery after novel perturbations, transfer, and graceful degradation?

S11 operationalized this as held-out trace-derived competency profiles. Primary outcome rule: `{json.dumps(dict(outcome), sort_keys=True)}`.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S11.
- Selected S08 policies: `{args.selected_policies}`.
- S09 memory-ablation finding source: `{args.s09_delta_summary}`.
- S10 communication-ablation finding source: `{args.s10_delta_summary}`.
- Prior finding summary: `{json.dumps(dict(prior_findings), sort_keys=True)}`.
- Datasets: none required.

## Methods

Implemented `src/e04/competency_metrics.py`, `tests/e04/test_competency_metrics.py`, and `scripts/e04_s11_competency_metrics.py`.

Policy modes:

- original/open-loop Bubble cell-view baseline.
- selected S08 discovered policies with the S09 no-memory mask.
- S09-supported neighbor memory with S10 no-signal control.
- S10 signaling sensitivity arms: nearest-neighbor, diffusive, long-range scalar, and noisy diffusive signaling.

S11 schedules use seeds starting at 41001 and custom `s11_` scenario names so they are separated from the S08 search grid and S09/S10 ablation grids. Metrics are computed offline from trace rows containing values, frozen positions, actions, route side, and event-level sortedness.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- matplotlib backend: Agg
- New dependencies installed: none.
- CPU use: serial stateful trajectory evaluation with worker count `1`; host logical CPUs recorded in run manifest.
- Platform: {platform.platform()}

## Parameters

- Protocol ID: `{S11_PROTOCOL_ID}`
- Selected policies: `{selected_policy_count}`
- Policy modes: `{list(REQUIRED_S11_POLICY_MODES)}`
- Competency axes: `{list(S11_COMPETENCY_AXES)}`
- Damage levels: `{list(S11_DEGRADATION_LEVELS)}`
- Max events: `{args.max_events}`
- Transfer max events: `{args.transfer_max_events}`
- Result table: `{result_path}`
- Trace table: `{trace_path}`
- Figure: `{figure_path}`
- Config: `{config_path}`

## Results

- Total result rows: `{len(df)}`
- Trace rows: `{int(df['trace_row_count'].sum())}`
- Policies evaluated: `{df['s08_source_policy_id'].nunique()}`
- Policy modes: `{df['policy_mode'].nunique()}`
- Competency axes: `{sorted(df['competency_axis'].unique())}`
- Oracle-hit rows: `{int(df['uses_global_oracle'].sum())}`
- Outcome: `{classification}`

### Competency Profiles

{summary_table}

### Matched Deltas

{delta_table}

### Graceful-Degradation Profile

{degradation_table}

## Validation Checks

{validation_table}

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S11 artifact generation if validation passed.
- The original/open-loop baseline is Bubble only, matched to the selected S08 Bubble-style adjacent-action policies; Selection and Insertion are not mixed into local-policy claims here.
- Competency metrics are bounded computational proxies from sorting-array traces, not causal evidence of cognition or biological repair.
- S11 does not revise S10's communication conclusion; signaling arms are included as sensitivity comparisons against the S09-supported memory-only mechanism.

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
- Trace parquet: `{trace_path}`
- Figure: `{figure_path}`

## Recommended Next Action

Hand control back to the Chief Scientist. If S11 is accepted, proceed to S12 tissue-field analysis with a train-test split by seed and perturbation.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_intelligence_like_competencies.parquet"
    trace_path = artifacts_dir / "traces" / "e04_s11_competency_traces.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_intelligence_like_competencies.csv"
    summary_path = artifacts_dir / "tables" / "e04_intelligence_like_competencies_summary.csv"
    delta_path = artifacts_dir / "tables" / "e04_intelligence_like_competencies_matched_deltas.csv"
    delta_summary_path = artifacts_dir / "tables" / "e04_intelligence_like_competencies_delta_summary.csv"
    degradation_path = artifacts_dir / "tables" / "e04_intelligence_like_competencies_degradation_profile.csv"
    validation_path = artifacts_dir / "tables" / "e04_intelligence_like_competencies_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "competency_profiles.png"
    config_path = artifacts_dir / "configs" / "e04_s11_competency_metrics.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_competency_metrics_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"

    configs = default_s11_eval_configs(max_events=args.max_events, transfer_max_events=args.transfer_max_events)
    modes = default_s11_policy_modes()
    design_audit = assert_s11_design(configs, modes)
    selected_policies = load_selected_s08_policies(args.selected_policies, limit=args.policy_limit)
    prior_findings = load_prior_findings(args)

    config_record = {
        "schema": "eidosoma.e04.s11.competency_metrics_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "protocolId": S11_PROTOCOL_ID,
        "selectedPoliciesPath": str(args.selected_policies),
        "selectedPolicyCount": len(selected_policies),
        "maxEvents": args.max_events,
        "transferMaxEvents": args.transfer_max_events,
        "workerCount": 1,
        "allowedTrainingSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedTrainingSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "policyModes": [mode.to_dict() for mode in modes],
        "evaluationConfigs": [config.to_dict() for config in configs],
        "matchedSeedDesignAudit": design_audit,
        "priorFindings": prior_findings,
    }
    write_json(config_path, config_record)

    rows, trace_rows = run_s11_competency_matrix(genomes=selected_policies, configs=configs, modes=modes)
    df = pd.DataFrame(rows).sort_values(
        ["s08_source_policy_id", "competency_axis", "scenario_name", "damage_level", "activation_seed", "policy_mode_order"]
    ).reset_index(drop=True)
    traces_df = pd.DataFrame(trace_rows).sort_values(["run_id", "event_step"]).reset_index(drop=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_path, index=False)
    traces_df.to_parquet(trace_path, index=False)
    df.to_csv(result_csv_path, index=False)

    summary_df = summarize_results(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    delta_df = matched_deltas(df)
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_df.to_csv(delta_path, index=False)
    delta_summary_df = summarize_deltas(delta_df)
    delta_summary_path.parent.mkdir(parents=True, exist_ok=True)
    delta_summary_df.to_csv(delta_summary_path, index=False)
    degradation_df = degradation_profile(df)
    degradation_path.parent.mkdir(parents=True, exist_ok=True)
    degradation_df.to_csv(degradation_path, index=False)
    outcome = infer_competency_outcome(delta_summary_df)
    write_figure(summary_df, degradation_df, figure_path)

    if args.run_unit_tests:
        e04_s11_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_competency_metrics"], args.repo_dir)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], args.repo_dir)
        e04_s10_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_communication_ablations"], args.repo_dir)
        e03_policy_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], args.repo_dir)
        e02_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], args.repo_dir)
    else:
        skipped = {"command": "skipped by --no-run-unit-tests", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
        e04_s11_tests = unit_tests = e04_s10_tests = e03_policy_tests = e02_tests = skipped

    validation_df = validate_results(
        df=df,
        traces_df=traces_df,
        delta_df=delta_df,
        configs=configs,
        modes=modes,
        selected_policy_count=len(selected_policies),
        design_audit=design_audit,
        prior_findings=prior_findings,
        unit_tests=unit_tests,
        e04_s11_tests=e04_s11_tests,
        e04_s10_tests=e04_s10_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)
    validation_success = bool(validation_df["success"].all())
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 S11 tests return code {e04_s11_tests['returnCode']}; E04 tests return code {unit_tests['returnCode']}; "
        f"E04 S10 tests return code {e04_s10_tests['returnCode']}; E03 policy tests return code {e03_policy_tests['returnCode']}; "
        f"E02 simulator tests return code {e02_tests['returnCode']}; {len(df)} competency rows; "
        f"trace rows={len(traces_df)}; oracle-hit rows={int(df['uses_global_oracle'].sum())}; "
        f"outcome={outcome.get('outcome')}"
    )

    source_files = [
        args.repo_dir / "src/e04/competency_metrics.py",
        args.repo_dir / "scripts/e04_s11_competency_metrics.py",
        args.repo_dir / "tests/e04/test_competency_metrics.py",
        args.repo_dir / "src/e04/communication_ablations.py",
        args.repo_dir / "src/e04/memory_ablations.py",
        args.repo_dir / "src/e04/evolutionary_search.py",
        args.repo_dir / "src/e04/no_oracle_protocol.py",
        args.repo_dir / "src/e04/homeostasis.py",
        args.repo_dir / "src/e04/memory_policies.py",
        args.repo_dir / "src/e04/signaling.py",
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
        trace_path,
        result_csv_path,
        summary_path,
        delta_path,
        delta_summary_path,
        degradation_path,
        validation_path,
        figure_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    report_text = render_report(
        df=df,
        summary_df=summary_df,
        delta_summary_df=delta_summary_df,
        degradation_df=degradation_df,
        validation_df=validation_df,
        outcome=outcome,
        prior_findings=prior_findings,
        validation_line=validation_line,
        validation_success=validation_success,
        artifact_paths=artifact_paths,
        result_path=result_path,
        trace_path=trace_path,
        figure_path=figure_path,
        config_path=config_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        args=args,
        selected_policy_count=len(selected_policies),
        unit_tests=unit_tests,
        e04_s11_tests=e04_s11_tests,
        e04_s10_tests=e04_s10_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    write_text(report_path, report_text)

    manifest_artifacts = [
        artifact_entry(report_path, artifacts_dir, "S11 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S11 competency result rows"),
        artifact_entry(trace_path, artifacts_dir, "S11 compact event traces used for offline metrics"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S11 competency rows"),
        artifact_entry(summary_path, artifacts_dir, "S11 competency profile summary table"),
        artifact_entry(delta_path, artifacts_dir, "S11 matched deltas versus open-loop and neighbor memory"),
        artifact_entry(delta_summary_path, artifacts_dir, "S11 matched delta summary table"),
        artifact_entry(degradation_path, artifacts_dir, "S11 graceful-degradation profile table"),
        artifact_entry(validation_path, artifacts_dir, "S11 validation table"),
        artifact_entry(figure_path, artifacts_dir, "S11 competency profile figure"),
        artifact_entry(config_path, artifacts_dir, "S11 competency configuration"),
        artifact_entry(source_manifest_path, artifacts_dir, "S11 source/provenance manifest"),
        manifest_self_entry(manifest_path, artifacts_dir, "S11 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S11 outputs"),
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
        "selectedPolicyCount": len(selected_policies),
        "resultRows": int(len(df)),
        "traceRows": int(len(traces_df)),
        "outcome": dict(outcome),
        "priorFindings": dict(prior_findings),
        "validation": validation_df.to_dict(orient="records"),
        "artifacts": manifest_artifacts,
    }
    write_json(run_manifest_path, run_manifest)

    checksum_lines = []
    for path in [
        report_path,
        result_path,
        trace_path,
        result_csv_path,
        summary_path,
        delta_path,
        delta_summary_path,
        degradation_path,
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
    print(f"traces={trace_path}")
    print(f"figure={figure_path}")
    return 0 if validation_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
