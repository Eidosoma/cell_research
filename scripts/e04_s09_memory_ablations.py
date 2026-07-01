#!/usr/bin/env python3
"""Run E04 S09 memory ablations for selected S08 local-only policies."""

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

from src.e04.memory_ablations import (
    REQUIRED_MEMORY_ABLATIONS,
    S09_PROTOCOL_ID,
    assert_matched_seed_design,
    default_memory_ablation_specs,
    default_s09_eval_configs,
    load_selected_s08_policies,
    run_memory_ablation_matrix,
)
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
)


STEP_ID = "S09"
STEP_NUMBER = 9
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--selected-policies", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "policies" / "e04_evolved_repair_policies.jsonl")
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


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["memory_ablation_order", "memory_ablation", "memory_ablation_label", "benchmark_family"], as_index=False)
        .agg(
            runs=("fitness_score", "size"),
            policies=("s08_source_policy_id", "nunique"),
            mean_fitness=("fitness_score", "mean"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_recovered_perturbation_count=("recovered_perturbation_count", "mean"),
            mean_unfreeze_count=("unfreeze_count", "mean"),
            mean_energy_total=("energy_total", "mean"),
            oracle_hit_rows=("uses_global_oracle", "sum"),
        )
        .sort_values(["memory_ablation_order", "benchmark_family"])
        .reset_index(drop=True)
    )
    for column in [
        "mean_fitness",
        "mean_time_in_target_fraction",
        "mean_final_sortedness_percent",
        "mean_recovered_perturbation_count",
        "mean_unfreeze_count",
        "mean_energy_total",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def matched_no_memory_deltas(df: pd.DataFrame) -> pd.DataFrame:
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
    base_cols = [
        *key_cols,
        "fitness_score",
        "time_in_target_fraction",
        "final_sortedness_percent",
        "recovered_perturbation_count",
        "unfreeze_count",
        "energy_total",
        "schedule_sha256",
    ]
    base = df[df["memory_ablation"] == "no_memory"][base_cols].rename(
        columns={
            "fitness_score": "no_memory_fitness_score",
            "time_in_target_fraction": "no_memory_time_in_target_fraction",
            "final_sortedness_percent": "no_memory_final_sortedness_percent",
            "recovered_perturbation_count": "no_memory_recovered_perturbation_count",
            "unfreeze_count": "no_memory_unfreeze_count",
            "energy_total": "no_memory_energy_total",
            "schedule_sha256": "no_memory_schedule_sha256",
        }
    )
    merged = df.merge(base, on=key_cols, how="left", validate="many_to_one")
    merged = merged[merged["memory_ablation"] != "no_memory"].copy()
    merged["fitness_delta_vs_no_memory"] = merged["fitness_score"] - merged["no_memory_fitness_score"]
    merged["time_in_target_delta_vs_no_memory"] = (
        merged["time_in_target_fraction"] - merged["no_memory_time_in_target_fraction"]
    )
    merged["final_sortedness_delta_vs_no_memory"] = (
        merged["final_sortedness_percent"] - merged["no_memory_final_sortedness_percent"]
    )
    merged["recovered_perturbation_delta_vs_no_memory"] = (
        merged["recovered_perturbation_count"] - merged["no_memory_recovered_perturbation_count"]
    )
    merged["unfreeze_delta_vs_no_memory"] = merged["unfreeze_count"] - merged["no_memory_unfreeze_count"]
    merged["energy_delta_vs_no_memory"] = merged["energy_total"] - merged["no_memory_energy_total"]
    merged["matched_schedule"] = merged["schedule_sha256"] == merged["no_memory_schedule_sha256"]
    return merged


def summarize_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    summary = (
        delta_df.groupby(["memory_ablation_order", "memory_ablation", "memory_ablation_label"], as_index=False)
        .agg(
            matched_pairs=("fitness_delta_vs_no_memory", "size"),
            mean_fitness_delta_vs_no_memory=("fitness_delta_vs_no_memory", "mean"),
            median_fitness_delta_vs_no_memory=("fitness_delta_vs_no_memory", "median"),
            positive_fitness_pair_fraction=("fitness_delta_vs_no_memory", lambda values: float((values > 0).mean())),
            mean_time_in_target_delta_vs_no_memory=("time_in_target_delta_vs_no_memory", "mean"),
            mean_final_sortedness_delta_vs_no_memory=("final_sortedness_delta_vs_no_memory", "mean"),
            mean_unfreeze_delta_vs_no_memory=("unfreeze_delta_vs_no_memory", "mean"),
            mean_energy_delta_vs_no_memory=("energy_delta_vs_no_memory", "mean"),
            matched_schedule_rows=("matched_schedule", "sum"),
        )
        .sort_values("memory_ablation_order")
        .reset_index(drop=True)
    )
    for column in [
        "mean_fitness_delta_vs_no_memory",
        "median_fitness_delta_vs_no_memory",
        "positive_fitness_pair_fraction",
        "mean_time_in_target_delta_vs_no_memory",
        "mean_final_sortedness_delta_vs_no_memory",
        "mean_unfreeze_delta_vs_no_memory",
        "mean_energy_delta_vs_no_memory",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def infer_smallest_reliable_memory(delta_summary_df: pd.DataFrame) -> dict[str, Any]:
    if delta_summary_df.empty:
        return {
            "smallestReliableMemory": "none",
            "criterion": "mean fitness delta > 0 and positive matched-pair fraction >= 0.60",
            "reason": "no matched no-memory deltas were available",
        }
    for record in delta_summary_df.sort_values("memory_ablation_order").to_dict(orient="records"):
        if (
            float(record["mean_fitness_delta_vs_no_memory"]) > 0.0
            and float(record["positive_fitness_pair_fraction"]) >= 0.60
        ):
            return {
                "smallestReliableMemory": str(record["memory_ablation"]),
                "label": str(record["memory_ablation_label"]),
                "criterion": "mean fitness delta > 0 and positive matched-pair fraction >= 0.60",
                "meanFitnessDelta": float(record["mean_fitness_delta_vs_no_memory"]),
                "positivePairFraction": float(record["positive_fitness_pair_fraction"]),
                "matchedPairs": int(record["matched_pairs"]),
            }
    return {
        "smallestReliableMemory": "none",
        "criterion": "mean fitness delta > 0 and positive matched-pair fraction >= 0.60",
        "reason": "no ablation met the predeclared reliability criterion",
    }


def write_figure(summary_df: pd.DataFrame, delta_summary_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    families = ["repair", "frozen_cell", "homeostatic"]
    for family in families:
        rows = summary_df[summary_df["benchmark_family"] == family].sort_values("memory_ablation_order")
        axes[0].plot(
            rows["memory_ablation_label"],
            rows["mean_fitness"],
            marker="o",
            label=family.replace("_", " "),
        )
    axes[0].set_title("Mean fitness by memory capacity")
    axes[0].set_ylabel("Mean fitness")
    axes[0].set_xlabel("Memory ablation")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    if not delta_summary_df.empty:
        rows = delta_summary_df.sort_values("memory_ablation_order")
        colors = ["#777777" if value <= 0 else "#2a7f62" for value in rows["mean_fitness_delta_vs_no_memory"]]
        axes[1].bar(rows["memory_ablation_label"], rows["mean_fitness_delta_vs_no_memory"], color=colors)
        axes[1].axhline(0.0, color="#333333", linewidth=1)
    axes[1].set_title("Matched delta versus no memory")
    axes[1].set_ylabel("Mean fitness delta")
    axes[1].set_xlabel("Memory ablation")
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
    unit_tests: Mapping[str, Any],
    e04_s09_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_rows = int(selected_policy_count * len(specs) * len(configs))
    rows.append(
        {
            "validation_case": "ablation_matrix_complete",
            "success": bool(len(df) == expected_rows and tuple(sorted(df["memory_ablation"].unique())) == tuple(sorted(REQUIRED_MEMORY_ABLATIONS))),
            "detail": f"rows={len(df)}; expected={expected_rows}; variants={sorted(df['memory_ablation'].unique())}",
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
    complete_groups = grouped["memory_ablation"].nunique().eq(len(REQUIRED_MEMORY_ABLATIONS)).all()
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
    capacity_records = [_parse_json(value) for value in df["memory_capacity_observed_json"]]
    capacity_violations = [
        violation
        for record in capacity_records
        for violation in record.get("capacityViolations", [])
    ]
    rows.append(
        {
            "validation_case": "memory_capacity_limits_enforced",
            "success": bool(not capacity_violations and all(record.get("recordCount", 0) > 0 for record in capacity_records)),
            "detail": f"capacity violation count={len(capacity_violations)}; first={capacity_violations[:5]}",
        }
    )
    feature_records = [_parse_json(value) for value in df["feature_audit_summary_json"]]
    rows.append(
        {
            "validation_case": "s07_projected_local_only_policy_inputs",
            "success": bool(
                (df["uses_global_oracle"] == False).all()
                and all(not record.get("usesGlobalOracle", True) for record in feature_records)
                and set(df["policy_input_contract"]) == {"S07 LocalTrainingObservation masked by S09 MemoryAblationSpec"}
            ),
            "detail": f"oracle rows={int(df['uses_global_oracle'].sum())}; protocol={LOCAL_ONLY_PROTOCOL_ID}",
        }
    )
    signal_records = [_parse_json(value) for value in df["signal_audit_json"]]
    forbidden_hits = [
        record
        for record in feature_records
        if record.get("forbiddenTokenHits") or record.get("excludedSignalHits") or record.get("listLikeFeatureValueHits")
    ]
    excluded_nonzero = [
        record
        for record in signal_records
        if not record.get("targetDerivedSignalFieldsZeroed", False)
    ]
    rows.append(
        {
            "validation_case": "target_derived_and_forbidden_fields_excluded",
            "success": bool(
                not forbidden_hits
                and not excluded_nonzero
                and set(ALLOWED_TRAINING_SIGNAL_FIELDS) == {"blocked", "frustrated"}
                and {"sorted", "target_seeking", "morphogen"}.issubset(set(EXCLUDED_TRAINING_SIGNAL_FIELDS))
            ),
            "detail": (
                f"feature forbidden/excluded hits={len(forbidden_hits)}; "
                f"target-derived nonzero signal rows={len(excluded_nonzero)}; "
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
            "validation_case": "selected_s08_policies_used",
            "success": bool(
                selected_policy_count > 0
                and df["s08_source_policy_id"].nunique() == selected_policy_count
                and (df["candidate_type"] == "s08_selected_memory_ablation").all()
            ),
            "detail": f"selected policies={selected_policy_count}; row policies={df['s08_source_policy_id'].nunique()}",
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
                and e04_s09_tests["success"]
                and e03_policy_tests["success"]
                and e02_tests["success"]
            ),
            "detail": (
                f"E04 discover={unit_tests['returnCode']}; E04 S09={e04_s09_tests['returnCode']}; "
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
    smallest: Mapping[str, Any],
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
    unit_tests: Mapping[str, Any],
    e04_s09_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> str:
    if not validation_success:
        outcome = "constraining/contradictory"
    elif smallest.get("smallestReliableMemory") == "none":
        outcome = "null"
    else:
        outcome = "supportive"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{e04_s09_tests['command']}` -> return code {e04_s09_tests['returnCode']}",
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_policy_tests['command']}` -> return code {e03_policy_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            (
                "- `python scripts/e04_s09_memory_ablations.py --repo-dir /workspace/cell-research "
                "--artifacts-dir $ARTIFACTS_DIR`"
            ),
        ]
    )
    summary_table = markdown_table(
        summary_df,
        [
            "memory_ablation_label",
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
            "memory_ablation_label",
            "matched_pairs",
            "mean_fitness_delta_vs_no_memory",
            "positive_fitness_pair_fraction",
            "mean_time_in_target_delta_vs_no_memory",
            "mean_final_sortedness_delta_vs_no_memory",
            "mean_unfreeze_delta_vs_no_memory",
        ],
        max_rows=12,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)
    smallest_text = json.dumps(dict(smallest), sort_keys=True)
    return f"""# E04 S09 Research Step Full Results

## Top Summary

- Research step ID: S09
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: S09 isolates policy-visible memory capacity for the two S08 selected adjacent-action policies under a compact matched-seed benchmark. It does not yet isolate communication mechanisms; signal-field memory is retained as a memory substrate and S10 must separately ablate communication range/noise/mode.
- Lay summary: The selected S08 local policies were rerun with progressively larger policy-visible memory: none, one last-success bit, bounded counters, bounded neighbor records, and a signal-field memory channel. Every ablation used the same seeds and perturbation schedules for direct paired comparison.
- Recommended next action: Hand control back to the Chief Scientist. If accepted, proceed to S10 communication ablations using the S09 memory findings as the matched memory baseline.

## Frozen Question

What is the smallest memory capacity that reliably improves robustness or repair?

S09 operationalized "reliably improves" as positive mean fitness delta versus no-memory and a positive matched-pair fraction of at least 0.60 across the compact matched repair, Frozen Cell, and homeostatic matrix. Result: `{smallest_text}`.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S09.
- Selected S08 policies: `{args.selected_policies}`.
- S07 local-only protocol: `src/e04/no_oracle_protocol.py`.
- S08 selected-policy implementation: `src/e04/evolutionary_search.py`.
- S05 task substrate: `src/e04/homeostasis.py`.
- S01/S02 interfaces: `src/e04/memory_policies.py`, `src/e04/signaling.py`.
- Datasets: none required.
- Previous mounted artifacts: E01 `/previous-artifacts/E01`, E02 `/previous-artifacts/E02`, E03 `/previous-artifacts/E03`.

## Methods

Implemented `src/e04/memory_ablations.py`, `tests/e04/test_memory_ablations.py`, and `scripts/e04_s09_memory_ablations.py`.

The S08 genomes were not retrained. For each action, the simulator first constructed the S07 `LocalTrainingObservation` and then applied an S09 `MemoryAblationSpec` mask before policy scoring. The policy-visible memory ladder was:

- `no_memory`: no cell memory and no persistent signal-field memory.
- `one_bit_memory`: only the last-move success bit.
- `bounded_counter_memory`: last success plus bounded time-since-movement, local frustration, and failed-swap counts.
- `neighbor_memory`: bounded counters plus two recent immediate-neighbor identity records.
- `signal_field_memory`: no cell-internal memory exposed; only allowed persistent `blocked` and `frustrated` signal fields.

Target-derived and forbidden policy inputs were excluded. `sorted`, `target_seeking`, and `morphogen` were not exposed to the policy, and S09 signal emission zeroed target-derived signal fields rather than computing them from `ideal_position`.

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

- Protocol ID: `{S09_PROTOCOL_ID}`
- Selected policies: `{selected_policy_count}`
- Memory ablations: `{list(REQUIRED_MEMORY_ABLATIONS)}`
- Evaluation configs: `{df.groupby(['benchmark_family', 'task_name']).ngroups}`
- Max events: `{args.max_events}`
- Held-out max events: `{args.heldout_max_events}`
- Result table: `{result_path}`
- Figure: `{figure_path}`
- Config: `{config_path}`

## Results

- Total result rows: `{len(df)}`
- Policies evaluated: `{df['s08_source_policy_id'].nunique()}`
- Memory variants: `{df['memory_ablation'].nunique()}`
- Benchmark families: `{sorted(df['benchmark_family'].unique())}`
- Oracle-hit rows: `{int(df['uses_global_oracle'].sum())}`
- Smallest reliable memory result: `{smallest.get('smallestReliableMemory')}`

### Summary By Memory And Task Family

{summary_table}

### Matched Deltas Versus No Memory

{delta_table}

## Validation Checks

{validation_table}

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S09 artifact generation if validation passed.
- S09 is a compact ablation of two selected S08 policies, not a broad proof over the full policy morphospace.
- The reliability criterion is an operational threshold over simulated fitness, not a biological claim.
- Memory and communication may interact non-additively. The `signal_field_memory` arm is informative but does not replace S10 communication ablations.
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

Hand control back to the Chief Scientist. If S09 is accepted, run S10 communication ablations with matched seeds and preserve the S07 local-only input contract.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_memory_ablations.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_memory_ablations.csv"
    summary_path = artifacts_dir / "tables" / "e04_memory_ablations_summary.csv"
    delta_path = artifacts_dir / "tables" / "e04_memory_ablations_matched_deltas.csv"
    delta_summary_path = artifacts_dir / "tables" / "e04_memory_ablations_delta_summary.csv"
    validation_path = artifacts_dir / "tables" / "e04_memory_ablations_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "memory_ablation_curves.png"
    config_path = artifacts_dir / "configs" / "e04_s09_memory_ablations.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_memory_ablations_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"

    specs = default_memory_ablation_specs()
    configs = default_s09_eval_configs(max_events=args.max_events, heldout_max_events=args.heldout_max_events)
    design_audit = assert_matched_seed_design(configs)
    selected_policies = load_selected_s08_policies(args.selected_policies, limit=args.policy_limit)

    config_record = {
        "schema": "eidosoma.e04.s09.memory_ablation_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "protocolId": S09_PROTOCOL_ID,
        "selectedPoliciesPath": str(args.selected_policies),
        "selectedPolicyCount": len(selected_policies),
        "maxEvents": args.max_events,
        "heldoutMaxEvents": args.heldout_max_events,
        "workerCount": 1,
        "allowedTrainingSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedTrainingSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "memoryAblations": [spec.to_dict() for spec in specs],
        "evaluationConfigs": [config.to_dict() for config in configs],
        "matchedSeedDesignAudit": design_audit,
    }
    write_json(config_path, config_record)

    rows = run_memory_ablation_matrix(genomes=selected_policies, configs=configs, specs=specs)
    df = pd.DataFrame(rows)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_path, index=False)
    df.to_csv(result_csv_path, index=False)

    summary_df = summarize_results(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    delta_df = matched_no_memory_deltas(df)
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_df.to_csv(delta_path, index=False)
    delta_summary_df = summarize_deltas(delta_df)
    delta_summary_path.parent.mkdir(parents=True, exist_ok=True)
    delta_summary_df.to_csv(delta_summary_path, index=False)
    smallest = infer_smallest_reliable_memory(delta_summary_df)
    write_figure(summary_df, delta_summary_df, figure_path)

    if args.run_unit_tests:
        e04_s09_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_memory_ablations"], args.repo_dir)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], args.repo_dir)
        e03_policy_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], args.repo_dir)
        e02_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], args.repo_dir)
    else:
        skipped = {"command": "skipped by --no-run-unit-tests", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
        e04_s09_tests = unit_tests = e03_policy_tests = e02_tests = skipped

    validation_df = validate_results(
        df=df,
        delta_df=delta_df,
        specs=specs,
        configs=configs,
        selected_policy_count=len(selected_policies),
        design_audit=design_audit,
        unit_tests=unit_tests,
        e04_s09_tests=e04_s09_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)
    validation_success = bool(validation_df["success"].all())
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 S09 tests return code {e04_s09_tests['returnCode']}; E04 tests return code {unit_tests['returnCode']}; "
        f"E03 policy tests return code {e03_policy_tests['returnCode']}; E02 simulator tests return code {e02_tests['returnCode']}; "
        f"{len(df)} ablation rows; oracle-hit rows={int(df['uses_global_oracle'].sum())}; "
        f"smallest reliable memory={smallest.get('smallestReliableMemory')}"
    )

    source_files = [
        args.repo_dir / "src/e04/memory_ablations.py",
        args.repo_dir / "scripts/e04_s09_memory_ablations.py",
        args.repo_dir / "tests/e04/test_memory_ablations.py",
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
        smallest=smallest,
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
        unit_tests=unit_tests,
        e04_s09_tests=e04_s09_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    write_text(report_path, report_text)

    manifest_artifacts = [
        artifact_entry(report_path, artifacts_dir, "S09 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S09 memory ablation result rows"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S09 memory ablation rows"),
        artifact_entry(summary_path, artifacts_dir, "S09 memory ablation summary table"),
        artifact_entry(delta_path, artifacts_dir, "S09 matched deltas versus no-memory rows"),
        artifact_entry(delta_summary_path, artifacts_dir, "S09 matched delta summary table"),
        artifact_entry(validation_path, artifacts_dir, "S09 validation table"),
        artifact_entry(figure_path, artifacts_dir, "S09 memory ablation curves figure"),
        artifact_entry(config_path, artifacts_dir, "S09 memory ablation configuration"),
        artifact_entry(source_manifest_path, artifacts_dir, "S09 source/provenance manifest"),
        manifest_self_entry(manifest_path, artifacts_dir, "S09 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S09 outputs"),
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
        "smallestReliableMemory": smallest,
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
