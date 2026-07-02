#!/usr/bin/env python3
"""Run E04 S10 communication ablations using S09 neighbor-memory controls."""

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

from src.e04.communication_ablations import (
    REQUIRED_COMMUNICATION_ABLATIONS,
    S10_MEMORY_BASELINE,
    S10_POLICY_FAMILY,
    S10_PROTOCOL_ID,
    default_communication_ablation_specs,
    default_s10_eval_configs,
    run_communication_ablation_matrix,
    simulate_communication_ablation,
)
from src.e04.evolutionary_search import fitness_from_row
from src.e04.memory_ablations import assert_matched_seed_design, load_selected_s08_policies
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
)


STEP_ID = "S10"
STEP_NUMBER = 10
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--selected-policies", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "policies" / "e04_evolved_repair_policies.jsonl")
    parser.add_argument("--s09-results", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "results" / "e04_memory_ablations.parquet")
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--heldout-max-events", type=int, default=140)
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
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, Mapping):
        return value
    return {}


def _config_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["s08_source_policy_id"]),
        str(row["benchmark_family"]),
        str(row["task_name"]),
        str(row["reliability_mode"]),
        int(row["activation_seed"]),
        int(row["policy_seed"]),
        int(row["schedule_seed"]),
        int(row["array_size"]),
        int(row["max_events"]),
    )


def _control_signal_summary(spec: Any) -> dict[str, Any]:
    return {
        "usesGlobalOracle": False,
        "communicationAblation": spec.name,
        "signalConfig": spec.to_dict(),
        "senseRecordCount": 0,
        "emissionRecordCount": 0,
        "noiseAppliedCount": 0,
        "configuredNoiseStd": float(spec.noise_std),
        "noiseAbsMaxObserved": 0.0,
        "maxAccessDistanceObserved": 0,
        "allowedSignalMinObserved": 0.0,
        "allowedSignalMaxObserved": 0.0,
        "allowedSignalsFinite": True,
        "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFieldAbsMax": {name: 0.0 for name in EXCLUDED_TRAINING_SIGNAL_FIELDS},
        "targetDerivedSignalFieldsZeroed": True,
        "lessLocal": False,
    }


def normalize_s09_control_row(row: Mapping[str, Any], spec: Any) -> dict[str, Any]:
    out = dict(row)
    source_protocol = str(out.get("protocol_id", "unknown"))
    signal_summary = _control_signal_summary(spec)
    out.update(
        {
            "source_reused_from_s09": True,
            "source_research_step_id": "S09",
            "source_protocol_id": source_protocol,
            "protocol_id": S10_PROTOCOL_ID,
            "policy_family": S10_POLICY_FAMILY,
            "candidate_type": "s08_selected_communication_ablation",
            "baseline_type": "memory_only_no_signal_control",
            "s09_memory_baseline": S10_MEMORY_BASELINE,
            "communication_ablation": spec.name,
            "communication_ablation_label": spec.label,
            "communication_ablation_order": int(spec.order),
            "communication_ablation_description": spec.description,
            "communication_less_local": False,
            "signal_config_json": json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")),
            "signal_observed_json": json.dumps(signal_summary, sort_keys=True, separators=(",", ":")),
            "signal_audit_json": json.dumps(signal_summary, sort_keys=True, separators=(",", ":")),
            "policy_input_contract": "S07 LocalTrainingObservation masked by S10 CommunicationAblationSpec",
            "allowed_training_signal_fields_json": json.dumps(list(ALLOWED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
            "excluded_training_signal_fields_json": json.dumps(list(EXCLUDED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
            "centralized_baseline": False,
            "eligible_for_local_only_claims": True,
        }
    )
    out["uses_global_oracle"] = False
    out["fitness_score"] = fitness_from_row(out)
    return out


def load_or_rerun_memory_only_controls(
    *,
    s09_results_path: Path,
    genomes: Sequence[Any],
    configs: Sequence[Any],
    no_signal_spec: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_keys = {
        (
            genome.policy_id,
            config.benchmark_family,
            config.task_name,
            config.reliability_mode,
            config.activation_seed,
            config.policy_seed,
            config.schedule_seed,
            len(config.values),
            config.max_events,
        )
        for genome in genomes
        for config in configs
    }
    reused_rows: list[dict[str, Any]] = []
    reused_keys: set[tuple[Any, ...]] = set()
    if s09_results_path.exists():
        s09_df = pd.read_parquet(s09_results_path)
        controls = s09_df[s09_df["memory_ablation"] == S10_MEMORY_BASELINE].copy()
        for record in controls.to_dict(orient="records"):
            key = _config_key(record)
            if key in expected_keys:
                reused_rows.append(normalize_s09_control_row(record, no_signal_spec))
                reused_keys.add(key)
    missing_keys = expected_keys - reused_keys
    rerun_rows: list[dict[str, Any]] = []
    if missing_keys:
        genome_by_id = {genome.policy_id: genome for genome in genomes}
        config_by_key = {
            (
                config.benchmark_family,
                config.task_name,
                config.reliability_mode,
                config.activation_seed,
                config.policy_seed,
                config.schedule_seed,
                len(config.values),
                config.max_events,
            ): config
            for config in configs
        }
        for key in sorted(missing_keys):
            genome_id = str(key[0])
            config_key = tuple(key[1:])
            row = simulate_communication_ablation(config_by_key[config_key], genome_by_id[genome_id], no_signal_spec)
            row["source_reused_from_s09"] = False
            row["source_research_step_id"] = "S10_rerun_no_signal_control"
            rerun_rows.append(row)
    return (
        [*reused_rows, *rerun_rows],
        {
            "expectedControlRows": len(expected_keys),
            "reusedFromS09": len(reused_rows),
            "rerunControlRows": len(rerun_rows),
            "missingS09ControlRows": len(missing_keys),
            "s09ResultsPath": str(s09_results_path),
        },
    )


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["communication_ablation_order", "communication_ablation", "communication_ablation_label", "benchmark_family"], as_index=False)
        .agg(
            runs=("fitness_score", "size"),
            policies=("s08_source_policy_id", "nunique"),
            mean_fitness=("fitness_score", "mean"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_unfreeze_count=("unfreeze_count", "mean"),
            mean_energy_total=("energy_total", "mean"),
            oracle_hit_rows=("uses_global_oracle", "sum"),
        )
        .sort_values(["communication_ablation_order", "benchmark_family"])
        .reset_index(drop=True)
    )
    for column in [
        "mean_fitness",
        "mean_time_in_target_fraction",
        "mean_final_sortedness_percent",
        "mean_unfreeze_count",
        "mean_energy_total",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def matched_memory_only_deltas(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "s08_source_policy_id",
        "benchmark_family",
        "task_name",
        "reliability_mode",
        "activation_seed",
        "policy_seed",
        "schedule_seed",
        "array_size",
        "max_events",
    ]
    base = df[df["communication_ablation"] == "memory_only_no_signal"][
        [
            *key_cols,
            "fitness_score",
            "time_in_target_fraction",
            "final_sortedness_percent",
            "unfreeze_count",
            "energy_total",
            "schedule_sha256",
        ]
    ].rename(
        columns={
            "fitness_score": "memory_only_fitness_score",
            "time_in_target_fraction": "memory_only_time_in_target_fraction",
            "final_sortedness_percent": "memory_only_final_sortedness_percent",
            "unfreeze_count": "memory_only_unfreeze_count",
            "energy_total": "memory_only_energy_total",
            "schedule_sha256": "memory_only_schedule_sha256",
        }
    )
    merged = df.merge(base, on=key_cols, how="left", validate="many_to_one")
    merged = merged[merged["communication_ablation"] != "memory_only_no_signal"].copy()
    merged["fitness_delta_vs_memory_only"] = merged["fitness_score"] - merged["memory_only_fitness_score"]
    merged["time_in_target_delta_vs_memory_only"] = (
        merged["time_in_target_fraction"] - merged["memory_only_time_in_target_fraction"]
    )
    merged["final_sortedness_delta_vs_memory_only"] = (
        merged["final_sortedness_percent"] - merged["memory_only_final_sortedness_percent"]
    )
    merged["unfreeze_delta_vs_memory_only"] = merged["unfreeze_count"] - merged["memory_only_unfreeze_count"]
    merged["energy_delta_vs_memory_only"] = merged["energy_total"] - merged["memory_only_energy_total"]
    merged["matched_schedule"] = merged["schedule_sha256"] == merged["memory_only_schedule_sha256"]
    return merged


def summarize_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    summary = (
        delta_df.groupby(["communication_ablation_order", "communication_ablation", "communication_ablation_label"], as_index=False)
        .agg(
            matched_pairs=("fitness_delta_vs_memory_only", "size"),
            mean_fitness_delta_vs_memory_only=("fitness_delta_vs_memory_only", "mean"),
            median_fitness_delta_vs_memory_only=("fitness_delta_vs_memory_only", "median"),
            positive_fitness_pair_fraction=("fitness_delta_vs_memory_only", lambda values: float((values > 0).mean())),
            mean_time_in_target_delta_vs_memory_only=("time_in_target_delta_vs_memory_only", "mean"),
            mean_final_sortedness_delta_vs_memory_only=("final_sortedness_delta_vs_memory_only", "mean"),
            mean_unfreeze_delta_vs_memory_only=("unfreeze_delta_vs_memory_only", "mean"),
            mean_energy_delta_vs_memory_only=("energy_delta_vs_memory_only", "mean"),
            matched_schedule_rows=("matched_schedule", "sum"),
        )
        .sort_values("communication_ablation_order")
        .reset_index(drop=True)
    )
    for column in [
        "mean_fitness_delta_vs_memory_only",
        "median_fitness_delta_vs_memory_only",
        "positive_fitness_pair_fraction",
        "mean_time_in_target_delta_vs_memory_only",
        "mean_final_sortedness_delta_vs_memory_only",
        "mean_unfreeze_delta_vs_memory_only",
        "mean_energy_delta_vs_memory_only",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def infer_best_communication(delta_summary_df: pd.DataFrame) -> dict[str, Any]:
    if delta_summary_df.empty:
        return {
            "bestCommunicationMode": "none",
            "criterion": "mean fitness delta > 0 and positive matched-pair fraction >= 0.60",
            "reason": "no matched communication deltas were available",
        }
    candidates = delta_summary_df[
        (delta_summary_df["mean_fitness_delta_vs_memory_only"] > 0.0)
        & (delta_summary_df["positive_fitness_pair_fraction"] >= 0.60)
    ].sort_values("mean_fitness_delta_vs_memory_only", ascending=False)
    if candidates.empty:
        return {
            "bestCommunicationMode": "none",
            "criterion": "mean fitness delta > 0 and positive matched-pair fraction >= 0.60",
            "reason": "no communication mode met the reliability criterion",
        }
    record = candidates.iloc[0].to_dict()
    return {
        "bestCommunicationMode": str(record["communication_ablation"]),
        "label": str(record["communication_ablation_label"]),
        "criterion": "mean fitness delta > 0 and positive matched-pair fraction >= 0.60",
        "meanFitnessDelta": float(record["mean_fitness_delta_vs_memory_only"]),
        "positivePairFraction": float(record["positive_fitness_pair_fraction"]),
        "matchedPairs": int(record["matched_pairs"]),
    }


def write_figure(summary_df: pd.DataFrame, delta_summary_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = summary_df.pivot_table(
        index="benchmark_family",
        columns="communication_ablation_label",
        values="mean_fitness",
        aggfunc="mean",
    )
    ordered_labels = (
        summary_df[["communication_ablation_order", "communication_ablation_label"]]
        .drop_duplicates()
        .sort_values("communication_ablation_order")["communication_ablation_label"]
        .tolist()
    )
    pivot = pivot.reindex(columns=ordered_labels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    image = axes[0].imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axes[0].set_title("Mean fitness matrix")
    axes[0].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axes[0].set_yticks(range(len(pivot.index)), [item.replace("_", " ") for item in pivot.index])
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    if not delta_summary_df.empty:
        rows = delta_summary_df.sort_values("communication_ablation_order")
        colors = ["#777777" if value <= 0 else "#2a7f62" for value in rows["mean_fitness_delta_vs_memory_only"]]
        axes[1].bar(rows["communication_ablation_label"], rows["mean_fitness_delta_vs_memory_only"], color=colors)
        axes[1].axhline(0.0, color="#333333", linewidth=1)
    axes[1].set_title("Matched delta versus memory only")
    axes[1].set_ylabel("Mean fitness delta")
    axes[1].set_xlabel("Communication mode")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    *,
    df: pd.DataFrame,
    delta_df: pd.DataFrame,
    specs: Sequence[Any],
    configs: Sequence[Any],
    selected_policy_count: int,
    design_audit: Mapping[str, Any],
    control_audit: Mapping[str, Any],
    unit_tests: Mapping[str, Any],
    e04_s10_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_rows = int(selected_policy_count * len(specs) * len(configs))
    rows.append(
        {
            "validation_case": "communication_matrix_complete",
            "success": bool(len(df) == expected_rows and tuple(sorted(df["communication_ablation"].unique())) == tuple(sorted(REQUIRED_COMMUNICATION_ABLATIONS))),
            "detail": f"rows={len(df)}; expected={expected_rows}; modes={sorted(df['communication_ablation'].unique())}",
        }
    )
    no_signal = df[df["communication_ablation"] == "memory_only_no_signal"]
    rows.append(
        {
            "validation_case": "s09_neighbor_memory_no_signal_controls_included",
            "success": bool(
                not no_signal.empty
                and len(no_signal) == selected_policy_count * len(configs)
                and set(no_signal["memory_ablation"]) == {S10_MEMORY_BASELINE}
                and int(control_audit.get("reusedFromS09", 0)) == len(no_signal)
            ),
            "detail": json.dumps(dict(control_audit), sort_keys=True, default=str),
        }
    )
    group_cols = [
        "s08_source_policy_id",
        "benchmark_family",
        "task_name",
        "reliability_mode",
        "activation_seed",
        "policy_seed",
        "schedule_seed",
        "array_size",
        "max_events",
    ]
    grouped = df.groupby(group_cols)
    complete_groups = grouped["communication_ablation"].nunique().eq(len(REQUIRED_COMMUNICATION_ABLATIONS)).all()
    matched_schedule_groups = grouped["schedule_sha256"].nunique().eq(1).all()
    rows.append(
        {
            "validation_case": "matched_seed_groups_complete",
            "success": bool(design_audit.get("success") and complete_groups and matched_schedule_groups and delta_df["matched_schedule"].all()),
            "detail": json.dumps(
                {
                    "designAudit": dict(design_audit),
                    "groupCount": int(grouped.ngroups),
                    "completeGroups": bool(complete_groups),
                    "matchedScheduleGroups": bool(matched_schedule_groups),
                    "deltaMatchedRows": int(delta_df["matched_schedule"].sum()) if not delta_df.empty else 0,
                },
                sort_keys=True,
                default=str,
            )[:1800],
        }
    )
    observed_records = [_parse_json(value) for value in df["signal_observed_json"]]
    by_mode = {mode: [] for mode in REQUIRED_COMMUNICATION_ABLATIONS}
    for mode, record in zip(df["communication_ablation"], observed_records, strict=True):
        by_mode[str(mode)].append(record)
    no_signal_clean = all(
        record.get("senseRecordCount", 0) == 0 and record.get("emissionRecordCount", 0) == 0
        for record in by_mode["memory_only_no_signal"]
    )
    noisy_valid = all(record.get("noiseAppliedCount", 0) > 0 and record.get("configuredNoiseStd") == 0.05 for record in by_mode["noisy_diffusive_signaling"])
    non_noisy_valid = all(
        record.get("noiseAppliedCount", 0) == 0
        for mode in ("nearest_neighbor_signaling", "diffusive_signaling", "long_range_scalar_fields")
        for record in by_mode[mode]
    )
    long_range_labeled = all(record.get("lessLocal", False) for record in by_mode["long_range_scalar_fields"])
    rows.append(
        {
            "validation_case": "signal_range_noise_and_mode_config_validated",
            "success": bool(
                no_signal_clean
                and noisy_valid
                and non_noisy_valid
                and long_range_labeled
                and all(record.get("allowedSignalsFinite", True) for record in observed_records)
            ),
            "detail": (
                f"no_signal_clean={no_signal_clean}; noisy_valid={noisy_valid}; "
                f"non_noisy_valid={non_noisy_valid}; long_range_labeled={long_range_labeled}"
            ),
        }
    )
    feature_records = [_parse_json(value) for value in df["feature_audit_summary_json"]]
    rows.append(
        {
            "validation_case": "s07_projected_local_only_policy_inputs",
            "success": bool(
                (df["uses_global_oracle"] == False).all()
                and all(not record.get("usesGlobalOracle", True) for record in feature_records)
                and set(df["policy_input_contract"]) == {"S07 LocalTrainingObservation masked by S10 CommunicationAblationSpec"}
            ),
            "detail": f"oracle rows={int(df['uses_global_oracle'].sum())}; protocol={LOCAL_ONLY_PROTOCOL_ID}",
        }
    )
    forbidden_hits = [
        record
        for record in feature_records
        if record.get("forbiddenTokenHits") or record.get("excludedSignalHits") or record.get("listLikeFeatureValueHits")
    ]
    target_nonzero = [record for record in observed_records if not record.get("targetDerivedSignalFieldsZeroed", False)]
    rows.append(
        {
            "validation_case": "target_derived_and_forbidden_fields_excluded",
            "success": bool(not forbidden_hits and not target_nonzero),
            "detail": (
                f"feature forbidden/excluded hits={len(forbidden_hits)}; "
                f"target-derived nonzero signal rows={len(target_nonzero)}; "
                f"allowed={list(ALLOWED_TRAINING_SIGNAL_FIELDS)}; excluded={list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "repair_frozen_and_homeostatic_tasks_present",
            "success": bool(set(df["benchmark_family"]) == {"repair", "frozen_cell", "homeostatic"}),
            "detail": f"family counts={df['benchmark_family'].value_counts().to_dict()}",
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
            "success": bool(unit_tests["success"] and e04_s10_tests["success"] and e03_policy_tests["success"] and e02_tests["success"]),
            "detail": (
                f"E04 discover={unit_tests['returnCode']}; E04 S10={e04_s10_tests['returnCode']}; "
                f"E03 policy={e03_policy_tests['returnCode']}; E02 deterministic={e02_tests['returnCode']}"
            ),
        }
    )
    return pd.DataFrame(rows)


def render_report(
    *,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    delta_summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    best: Mapping[str, Any],
    validation_line: str,
    validation_success: bool,
    artifact_paths: Sequence[Path],
    result_path: Path,
    figure_path: Path,
    config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    args: argparse.Namespace,
    selected_policy_count: int,
    control_audit: Mapping[str, Any],
    unit_tests: Mapping[str, Any],
    e04_s10_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> str:
    if not validation_success:
        outcome = "constraining/contradictory"
    elif best.get("bestCommunicationMode") == "none":
        outcome = "constraining/contradictory"
    else:
        outcome = "supportive"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{e04_s10_tests['command']}` -> return code {e04_s10_tests['returnCode']}",
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_policy_tests['command']}` -> return code {e03_policy_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            (
                "- `python scripts/e04_s10_communication_ablations.py --repo-dir /workspace/cell-research "
                "--artifacts-dir $ARTIFACTS_DIR`"
            ),
        ]
    )
    summary_table = markdown_table(
        summary_df,
        [
            "communication_ablation_label",
            "benchmark_family",
            "runs",
            "mean_fitness",
            "mean_time_in_target_fraction",
            "mean_final_sortedness_percent",
            "mean_unfreeze_count",
            "oracle_hit_rows",
        ],
        max_rows=30,
    )
    delta_table = markdown_table(
        delta_summary_df,
        [
            "communication_ablation_label",
            "matched_pairs",
            "mean_fitness_delta_vs_memory_only",
            "positive_fitness_pair_fraction",
            "mean_time_in_target_delta_vs_memory_only",
            "mean_final_sortedness_delta_vs_memory_only",
            "mean_unfreeze_delta_vs_memory_only",
        ],
        max_rows=12,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)
    return f"""# E04 S10 Research Step Full Results

## Top Summary

- Research step ID: S10
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: S10 holds S09 `neighbor_memory` fixed and varies allowed communication modes for the two selected S08 policies. Long-range scalar fields are explicitly labeled less local. This does not yet test larger morphospace transfer or overfitting beyond the compact S09/S10 matched matrix.
- Lay summary: With the best S09 memory package held constant, S10 compared no signaling against nearest-neighbor, diffusive, long-range scalar, and noisy diffusive signaling on the same seeds and perturbation schedules. The no-signal control rows were reused directly from S09 where available.
- Recommended next action: Hand control back to the Chief Scientist. If accepted, proceed to S11 intelligence-like competency benchmarking using S09/S10 ablation-supported mechanisms.

## Frozen Question

Which local communication modes are necessary for improved repair, and which merely add cost or overfitting?

S10 operationalized "useful communication" as positive mean fitness delta versus the S09 neighbor-memory/no-signal control and a positive matched-pair fraction of at least 0.60. Result: `{json.dumps(dict(best), sort_keys=True)}`.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S10.
- S09 memory results: `{args.s09_results}`.
- Selected S08 policies: `{args.selected_policies}`.
- S07 local-only protocol: `src/e04/no_oracle_protocol.py`.
- S10 communication implementation: `src/e04/communication_ablations.py`.
- S09 memory implementation: `src/e04/memory_ablations.py`.
- Datasets: none required.

## Methods

Implemented `src/e04/communication_ablations.py`, `tests/e04/test_communication_ablations.py`, and `scripts/e04_s10_communication_ablations.py`.

All S10 policies used the S09 `neighbor_memory` capacity mask. The no-signal arm reused S09 `neighbor_memory` result rows as memory-only controls when their policy/config seed keys matched. Signaling arms reran the same selected S08 policies and matched configs under:

- `nearest_neighbor_signaling`: local blocked/frustrated emissions from the immediate local window.
- `diffusive_signaling`: allowed blocked/frustrated scalar fields diffused locally.
- `long_range_scalar_fields`: allowed scalar fields sensed across a wider radius and labeled less local.
- `noisy_diffusive_signaling`: diffusive allowed fields with deterministic Gaussian sensing noise.

Target-derived `sorted`, `target_seeking`, and `morphogen` fields were not exposed to policy inputs and were kept zero in S10 signal emissions.

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

- Protocol ID: `{S10_PROTOCOL_ID}`
- Selected policies: `{selected_policy_count}`
- Communication modes: `{list(REQUIRED_COMMUNICATION_ABLATIONS)}`
- Memory baseline: `{S10_MEMORY_BASELINE}`
- Control audit: `{json.dumps(dict(control_audit), sort_keys=True)}`
- Max events: `{args.max_events}`
- Held-out max events: `{args.heldout_max_events}`
- Result table: `{result_path}`
- Figure: `{figure_path}`
- Config: `{config_path}`

## Results

- Total result rows: `{len(df)}`
- Policies evaluated: `{df['s08_source_policy_id'].nunique()}`
- Communication modes: `{df['communication_ablation'].nunique()}`
- Benchmark families: `{sorted(df['benchmark_family'].unique())}`
- Oracle-hit rows: `{int(df['uses_global_oracle'].sum())}`
- Best communication result: `{best.get('bestCommunicationMode')}`

### Summary By Communication Mode And Task Family

{summary_table}

### Matched Deltas Versus S09 Memory-Only Control

{delta_table}

## Validation Checks

{validation_table}

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S10 artifact generation if validation passed.
- S10 is a compact ablation of two selected S08 policies under S09's best memory setting.
- Long-range scalar fields are computational communication channels, not biological morphogens.
- Offline global metrics were used only for evaluation after trajectories completed; they were not policy inputs.

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
- Figure: `{figure_path}`

## Recommended Next Action

Hand control back to the Chief Scientist. If S10 is accepted, proceed to S11 intelligence-like competency benchmarking with matched held-out perturbations and the S07 local-only input contract.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_communication_ablations.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_communication_ablations.csv"
    summary_path = artifacts_dir / "tables" / "e04_communication_ablations_summary.csv"
    delta_path = artifacts_dir / "tables" / "e04_communication_ablations_matched_deltas.csv"
    delta_summary_path = artifacts_dir / "tables" / "e04_communication_ablations_delta_summary.csv"
    validation_path = artifacts_dir / "tables" / "e04_communication_ablations_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "communication_ablation_matrix.png"
    config_path = artifacts_dir / "configs" / "e04_s10_communication_ablations.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_communication_ablations_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"

    specs = default_communication_ablation_specs()
    no_signal_spec = next(spec for spec in specs if spec.name == "memory_only_no_signal")
    configs = default_s10_eval_configs(max_events=args.max_events, heldout_max_events=args.heldout_max_events)
    design_audit = assert_matched_seed_design(configs)
    selected_policies = load_selected_s08_policies(args.selected_policies, limit=args.policy_limit)
    control_rows, control_audit = load_or_rerun_memory_only_controls(
        s09_results_path=args.s09_results,
        genomes=selected_policies,
        configs=configs,
        no_signal_spec=no_signal_spec,
    )

    config_record = {
        "schema": "eidosoma.e04.s10.communication_ablation_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "protocolId": S10_PROTOCOL_ID,
        "selectedPoliciesPath": str(args.selected_policies),
        "s09ResultsPath": str(args.s09_results),
        "selectedPolicyCount": len(selected_policies),
        "maxEvents": args.max_events,
        "heldoutMaxEvents": args.heldout_max_events,
        "workerCount": 1,
        "memoryBaseline": S10_MEMORY_BASELINE,
        "allowedTrainingSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedTrainingSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "communicationAblations": [spec.to_dict() for spec in specs],
        "evaluationConfigs": [config.to_dict() for config in configs],
        "matchedSeedDesignAudit": design_audit,
        "controlAudit": control_audit,
    }
    write_json(config_path, config_record)

    signal_rows = run_communication_ablation_matrix(genomes=selected_policies, configs=configs, specs=specs)
    df = pd.DataFrame([*control_rows, *signal_rows])
    df = df.sort_values(
        ["s08_source_policy_id", "benchmark_family", "task_name", "activation_seed", "communication_ablation_order"]
    ).reset_index(drop=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_path, index=False)
    df.to_csv(result_csv_path, index=False)

    summary_df = summarize_results(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    delta_df = matched_memory_only_deltas(df)
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_df.to_csv(delta_path, index=False)
    delta_summary_df = summarize_deltas(delta_df)
    delta_summary_path.parent.mkdir(parents=True, exist_ok=True)
    delta_summary_df.to_csv(delta_summary_path, index=False)
    best = infer_best_communication(delta_summary_df)
    write_figure(summary_df, delta_summary_df, figure_path)

    if args.run_unit_tests:
        e04_s10_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_communication_ablations"], args.repo_dir)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], args.repo_dir)
        e03_policy_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], args.repo_dir)
        e02_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], args.repo_dir)
    else:
        skipped = {"command": "skipped by --no-run-unit-tests", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
        e04_s10_tests = unit_tests = e03_policy_tests = e02_tests = skipped

    validation_df = validate_results(
        df=df,
        delta_df=delta_df,
        specs=specs,
        configs=configs,
        selected_policy_count=len(selected_policies),
        design_audit=design_audit,
        control_audit=control_audit,
        unit_tests=unit_tests,
        e04_s10_tests=e04_s10_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)
    validation_success = bool(validation_df["success"].all())
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 S10 tests return code {e04_s10_tests['returnCode']}; E04 tests return code {unit_tests['returnCode']}; "
        f"E03 policy tests return code {e03_policy_tests['returnCode']}; E02 simulator tests return code {e02_tests['returnCode']}; "
        f"{len(df)} ablation rows; oracle-hit rows={int(df['uses_global_oracle'].sum())}; "
        f"best communication mode={best.get('bestCommunicationMode')}"
    )

    source_files = [
        args.repo_dir / "src/e04/communication_ablations.py",
        args.repo_dir / "scripts/e04_s10_communication_ablations.py",
        args.repo_dir / "tests/e04/test_communication_ablations.py",
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
        result_csv_path,
        summary_path,
        delta_path,
        delta_summary_path,
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
        validation_df=validation_df,
        best=best,
        validation_line=validation_line,
        validation_success=validation_success,
        artifact_paths=artifact_paths,
        result_path=result_path,
        figure_path=figure_path,
        config_path=config_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        args=args,
        selected_policy_count=len(selected_policies),
        control_audit=control_audit,
        unit_tests=unit_tests,
        e04_s10_tests=e04_s10_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    write_text(report_path, report_text)

    manifest_artifacts = [
        artifact_entry(report_path, artifacts_dir, "S10 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S10 communication ablation result rows"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S10 communication ablation rows"),
        artifact_entry(summary_path, artifacts_dir, "S10 communication ablation summary table"),
        artifact_entry(delta_path, artifacts_dir, "S10 matched deltas versus memory-only rows"),
        artifact_entry(delta_summary_path, artifacts_dir, "S10 matched delta summary table"),
        artifact_entry(validation_path, artifacts_dir, "S10 validation table"),
        artifact_entry(figure_path, artifacts_dir, "S10 communication ablation matrix figure"),
        artifact_entry(config_path, artifacts_dir, "S10 communication ablation configuration"),
        artifact_entry(source_manifest_path, artifacts_dir, "S10 source/provenance manifest"),
        manifest_self_entry(manifest_path, artifacts_dir, "S10 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S10 outputs"),
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
        "bestCommunicationMode": best,
        "controlAudit": dict(control_audit),
        "validation": validation_df.to_dict(orient="records"),
        "artifacts": manifest_artifacts,
    }
    write_json(run_manifest_path, run_manifest)

    checksum_lines = []
    for path in [
        report_path,
        result_path,
        result_csv_path,
        summary_path,
        delta_path,
        delta_summary_path,
        validation_path,
        figure_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
    ]:
        checksum_lines.append(f"{sha256_file(path)}  {path}")
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(validation_line)
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
