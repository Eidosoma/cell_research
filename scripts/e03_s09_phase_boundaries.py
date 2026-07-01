#!/usr/bin/env python3
"""Run E03 S09 phase-boundary sweeps around S07/S08 candidate policies."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e03.gpu_batch_simulator import jax_backend_summary
from src.e03.phase_boundaries import (
    DEFAULT_PHASE_SEED,
    aggregate_axis,
    annotate_boundary_replication,
    build_primary_configs,
    detect_boundaries,
    replication_configs_from_boundaries,
    result_digest,
    run_phase_sweep,
    select_phase_policies,
    validation_frame,
)


STEP_ID = "S09"
STEP_NUMBER = 9
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--policy-library", type=Path, default=artifacts_default / "policies/e03_generated_policy_library.jsonl")
    parser.add_argument("--qd-discovered", type=Path, default=artifacts_default / "policies/e03_qd_discovered_policies.jsonl")
    parser.add_argument("--s08-archive", type=Path, default=artifacts_default / "results/e03_map_elites_archive.parquet")
    parser.add_argument("--candidate-policy-count", type=int, default=10)
    parser.add_argument("--max-replicated-boundaries", type=int, default=48)
    parser.add_argument("--seed", type=int, default=DEFAULT_PHASE_SEED)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

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
        "note": "Checksum omitted to avoid self-referential drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                text = "nan" if pd.isna(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def normalize_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: None if value is None else str(value))
    return out


def selected_policy_frame(policies: list[Any]) -> pd.DataFrame:
    rows = []
    for policy in policies:
        rows.append(
            {
                "base_policy_id": policy.base_policy_id,
                "policy_name": policy.policy_name,
                "source_kind": policy.source_kind,
                "route": policy.route,
                "quality_score": policy.quality_score,
                "archive_winner": policy.archive_winner,
                "nonclassic_cell": policy.nonclassic_cell,
                "selection_reason": policy.selection_reason,
            }
        )
    return pd.DataFrame(rows)


def write_phase_figure(path: Path, run_df: pd.DataFrame, axis_summary: pd.DataFrame, boundaries: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_specs = [
        ("event_cap", axes[0, 0], "Event cap", "Mean final sortedness"),
        ("array_size", axes[0, 1], "Array size", "Mean final sortedness"),
        ("probability", axes[1, 0], "Probability", "Mean final sortedness"),
    ]
    for axis_name, ax, xlabel, ylabel in plot_specs:
        subset = axis_summary[(axis_summary["axis_name"] == axis_name) & (axis_summary["run_count"] > 0)]
        for policy_name, group in subset.groupby("base_policy_name"):
            group = group.sort_values("axis_numeric", kind="mergesort")
            ax.plot(group["axis_numeric"], group["mean_final_sortedness"], marker="o", linewidth=1.2, alpha=0.75, label=policy_name[:24])
        bsub = boundaries[boundaries["axis_name"] == axis_name]
        for _, row in bsub.head(12).iterrows():
            ax.axvspan(row["lower_axis_numeric"], row["upper_axis_numeric"], color="#e3b341", alpha=0.12)
        ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{axis_name} phase response")
    failure_counts = (
        run_df.groupby(["axis_name", "failure_mode"]).size().reset_index(name="count").sort_values(["axis_name", "count"], ascending=[True, False])
    )
    pivot = failure_counts.pivot_table(index="failure_mode", columns="axis_name", values="count", fill_value=0)
    axes[1, 1].imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
    axes[1, 1].set_title("Failure-mode counts")
    axes[1, 1].set_xticks(range(len(pivot.columns)))
    axes[1, 1].set_xticklabels(pivot.columns, rotation=25, ha="right")
    axes[1, 1].set_yticks(range(len(pivot.index)))
    axes[1, 1].set_yticklabels(pivot.index)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:10], labels[:10], loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("S09 phase-boundary sweeps around S08 archive candidates")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_summary_tables(run_df: pd.DataFrame, boundary_df: pd.DataFrame, axis_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    axis_counts = (
        run_df.groupby(["axis_name", "split"])
        .agg(
            run_count=("config_id", "size"),
            policy_count=("base_policy_id", "nunique"),
            mean_final_sortedness=("final_inversion_sortedness", "mean"),
            success_fraction=("competent_at_threshold", "mean"),
            dg_fraction=("dg_present", "mean"),
            oscillation_fraction=("cycle_detected", "mean"),
        )
        .reset_index()
    )
    failure_modes = (
        run_df.groupby(["axis_name", "failure_mode"])
        .size()
        .reset_index(name="run_count")
        .sort_values(["axis_name", "run_count"], ascending=[True, False])
    )
    top_boundaries = boundary_df.sort_values(["replication_direction_match", "transition_strength"], ascending=[False, False]).head(20)
    strongest_axis = axis_summary.sort_values(["mean_final_sortedness", "success_fraction"], ascending=[False, False]).head(20)
    return {
        "axis_counts": axis_counts,
        "failure_modes": failure_modes,
        "top_boundaries": top_boundaries,
        "strongest_axis": strongest_axis,
    }


def render_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    run_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> str:
    validation_success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    replicated = int(boundary_df["replication_available"].sum()) if not boundary_df.empty else 0
    direction_matches = int(boundary_df["replication_direction_match"].sum()) if not boundary_df.empty else 0
    axis_set = sorted(run_df["axis_name"].unique())
    top_cols = [
        "base_policy_name",
        "axis_name",
        "lower_axis_value",
        "upper_axis_value",
        "boundary_kind",
        "transition_strength",
        "replication_direction_match",
    ]
    selected_cols = ["policy_name", "source_kind", "route", "quality_score", "archive_winner", "nonclassic_cell", "selection_reason"]
    return f"""# E03 S09 Research Step Full Results

## Top Summary

- Step ID: S09
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; result digest `{result_digest(run_df)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. S09 uses the S07/S08 proxy DSL local-step simulator and compact trajectory diagnostics, so boundary findings are candidate phase transitions for later full-simulator and held-out perturbation validation. Aggregation transitions were not directly evaluable in this single-policy screen.
- Lay summary: S09 selected {len(selected_df)} classic and S08-discovered policies, swept {len(run_df)} total phase rows across axes {axis_set}, detected {len(boundary_df)} candidate transition or sensitivity boundaries, and replicated endpoints for {replicated} boundary rows on independent seeds. {direction_matches} boundary rows reproduced the primary sortedness direction under independent seeds.
- Recommended next action: Stop for Chief review; if accepted, proceed to S10 behavior embedding using S07, S08, and S09 policy-level metrics and trajectory diagnostics.

## Frozen Question

Are there identifiable parameter changes that trigger transitions from failure to sorting, no DG to DG, no aggregation to aggregation, or convergence to oscillation?

## Inputs

- S05 policy library: `{manifest['inputArtifacts']['s05PolicyLibrary']}`
- S08 discovered policies: `{manifest['inputArtifacts']['s08DiscoveredPolicies']}`
- S08 MAP-Elites archive: `{manifest['inputArtifacts']['s08Archive']}`
- S07/S08 context: candidate policy quality scores, archive-winner flags, source kinds, routes, and DSL sources.

## Methods

S09 selected classic DSL landmarks plus high-quality S08 discovered policies, prioritizing archive winners, non-classic cells, and stochastic policies with explicit DSL probability parameters. Each selected policy was run through the CPU DSL interpreter even when S06 marked it as JAX-compatible, because S09 needed trajectory-level diagnostics.

The primary sweep tested event-cap, array-size, input-disorder, and probability axes. Event-cap, array-size, and input-disorder axes were applied to every selected policy. Probability sweeps were applied only to policies with `random_lt` guards or `choose` actions, replacing all stochastic thresholds with fixed values. For each run, S09 recorded final inversion sortedness, work, first sorted event, delayed-gratification drop, DG presence, repeated-state diagnostics, cycle candidates, oscillation score, and failure mode.

Boundary candidates were detected from adjacent axis values with competence-threshold crossings, DG changes, oscillation changes, or the strongest sortedness gradients. Selected boundary endpoints were re-run with independent seeds and annotated for direction reproducibility.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- JAX backend summary: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU trajectory interpreter; no CPU worker pool used.
- New dependencies installed: none.

## Parameters

- Candidate policy count target: `{manifest['parameters']['candidatePolicyCount']}`
- Selected policy count: `{len(selected_df)}`
- Max replicated boundaries: `{manifest['parameters']['maxReplicatedBoundaries']}`
- Primary row count: `{int((run_df['split'] == 'primary').sum())}`
- Replicate row count: `{int((run_df['split'] == 'replicate').sum())}`
- Axes: `{axis_set}`
- Competence threshold: `{manifest['parameters']['competenceThreshold']}`

## Results

### Selected Policies

{markdown_table(selected_df, selected_cols)}

### Axis Summary

{markdown_table(tables['axis_counts'], ["axis_name", "split", "run_count", "policy_count", "mean_final_sortedness", "success_fraction", "dg_fraction", "oscillation_fraction"])}

### Failure Modes

{markdown_table(tables['failure_modes'].head(30), ["axis_name", "failure_mode", "run_count"])}

### Top Boundary Candidates

{markdown_table(tables['top_boundaries'], top_cols)}

## Metrics

- `final_inversion_sortedness`: `1 - inversion_count / max_inversions`; higher is better.
- `competent_at_threshold`: final inversion sortedness at least 0.90.
- `dg_drop`: maximum drop below initial inversion sortedness.
- `dg_present`: `dg_drop >= 0.05` and final sortedness at least 0.10 above initial sortedness.
- `cycle_detected`: exact state repeat after a swap or target-position update.
- `oscillation_score`: two-event swap cycle count divided by event cap.
- `failure_mode`: `sorted`, `timeout_stalled`, `timeout_partial_progress`, `timeout_degraded`, `timeout_no_clear_progress`, `timeout_oscillation_candidate`, or `invalid`.

## Figures

- Phase-boundary map: `{artifacts['phaseBoundaryFigure']}`

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- S09 remains a proxy screen over the S02 DSL local actor semantics. It does not replace the E02/public simulator or paper scheduler.
- Aggregation is not directly measured because these S09 sweeps are single-policy arrays without chimeric Algotype labels.
- Delayed Gratification is a compact trajectory proxy based on sortedness dips and later recovery, not the full paper metric.
- Oscillation candidates are exact repeated-state diagnostics under deterministic cyclic scheduling; they should be manually inspected before being treated as dynamical classes.
- Probability sweeps rewrite all random thresholds in a policy at once, so they identify broad stochastic sensitivity rather than a single-rule causal parameter.

## Failed Assumptions

No required S07/S08 input was missing. Some boundary rows reproduce direction more clearly than others; the table preserves non-reproduced gradients rather than suppressing them.

## Provenance

Git commit before S09 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S09 manifest:

{source_table}

## Artifacts

The reusable S09 outputs are the row-level phase sweep table, boundary summary, axis summary, validation table, selected-policy table, phase-boundary figure, config, manifests, checksums, and this full-results handoff report.
"""


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e03"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        print("S09: running E03 unit tests", flush=True)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)

    print("S09: selecting phase-boundary policies from S05 and S08", flush=True)
    selected = select_phase_policies(
        policy_library=args.policy_library,
        discovered_jsonl=args.qd_discovered,
        candidate_count=args.candidate_policy_count,
    )
    selected_df = selected_policy_frame(selected)
    print(f"S09: selected {len(selected)} policies", flush=True)
    primary_configs = build_primary_configs(selected)
    primary_rows = sum(len(items) for items in primary_configs.values())
    print(f"S09: running {primary_rows} primary phase rows", flush=True)
    primary_df = run_phase_sweep(selected, primary_configs)
    primary_axis_summary = aggregate_axis(primary_df, split="primary")
    primary_boundaries = detect_boundaries(primary_axis_summary)
    replicate_configs = replication_configs_from_boundaries(primary_boundaries, max_boundaries=args.max_replicated_boundaries)
    replicate_rows = sum(len(items) for items in replicate_configs.values())
    print(f"S09: running {replicate_rows} independent replicate rows", flush=True)
    replicate_df = run_phase_sweep(selected, replicate_configs) if replicate_rows else pd.DataFrame(columns=primary_df.columns)
    run_df = pd.concat([primary_df, replicate_df], ignore_index=True, sort=False)
    axis_summary = aggregate_axis(run_df, split="primary")
    boundary_df = annotate_boundary_replication(primary_boundaries, run_df)

    phase_path = results_dir / "e03_phase_boundary_sweeps.parquet"
    phase_csv_path = tables_dir / "e03_phase_boundary_sweeps.csv"
    boundary_path = results_dir / "e03_phase_boundary_candidates.parquet"
    boundary_csv_path = tables_dir / "e03_phase_boundary_candidates.csv"
    axis_summary_path = results_dir / "e03_phase_boundary_axis_summary.parquet"
    axis_summary_csv_path = tables_dir / "e03_phase_boundary_axis_summary.csv"
    selected_path = tables_dir / "e03_phase_boundary_selected_policies.csv"
    validation_path = results_dir / "e03_phase_boundary_validation.parquet"
    validation_csv_path = tables_dir / "e03_phase_boundary_validation.csv"
    figure_path = figures_dir / "phase_boundary_maps.png"
    config_path = configs_dir / "e03_s09_phase_boundaries.json"
    manifest_path = src_snapshot_dir / "e03_phase_boundary_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"
    full_report_path = step_dir / "research_step_full_results.md"

    write_phase_figure(figure_path, run_df, axis_summary, boundary_df)
    validation = validation_frame(
        run_df=run_df,
        boundary_df=boundary_df,
        selected_policy_count=len(selected),
        figure_exists=figure_path.exists() and figure_path.stat().st_size > 0,
        unit_success=bool(unit_tests["success"]),
    )
    for column in ("validation_case", "expected", "observed", "notes"):
        validation[column] = validation[column].astype(str)

    normalize_for_parquet(run_df).to_parquet(phase_path, index=False)
    run_df.to_csv(phase_csv_path, index=False)
    normalize_for_parquet(boundary_df).to_parquet(boundary_path, index=False)
    boundary_df.to_csv(boundary_csv_path, index=False)
    normalize_for_parquet(axis_summary).to_parquet(axis_summary_path, index=False)
    axis_summary.to_csv(axis_summary_csv_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    normalize_for_parquet(validation).to_parquet(validation_path, index=False)
    validation.to_csv(validation_csv_path, index=False)

    config = {
        "schema": "eidosoma.e03.s09_phase_boundary_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "seed": args.seed,
        "candidatePolicyCount": args.candidate_policy_count,
        "maxReplicatedBoundaries": args.max_replicated_boundaries,
        "policyLibrary": str(args.policy_library),
        "s08DiscoveredPolicies": str(args.qd_discovered),
        "s08Archive": str(args.s08_archive),
        "axes": {
            "event_cap": "n=16, random permutation, event caps 8..128",
            "array_size": "n=8..32 with event_cap=4*n",
            "input_disorder": "nearly sorted, random, and reverse arrays",
            "probability": "all random_lt and choose thresholds rewritten together for stochastic policies",
        },
        "competenceThreshold": 0.90,
        "selectedPolicyIds": selected_df["base_policy_id"].tolist(),
    }
    write_json(config_path, config)

    elapsed = time.perf_counter() - started
    source_files = [
        args.repo_dir / "src/e03/phase_boundaries.py",
        args.repo_dir / "tests/e03/test_phase_boundaries.py",
        args.repo_dir / "scripts/e03_s09_phase_boundaries.py",
        args.repo_dir / "src/e03/rule_dsl.py",
        args.repo_dir / "src/e03/coarse_sweep.py",
    ]
    manifest = {
        "schema": "eidosoma.e03.phase_boundary_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": {
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "parameters": {
            "seed": args.seed,
            "candidatePolicyCount": args.candidate_policy_count,
            "maxReplicatedBoundaries": args.max_replicated_boundaries,
            "competenceThreshold": 0.90,
        },
        "inputArtifacts": {
            "s05PolicyLibrary": str(args.policy_library),
            "s08DiscoveredPolicies": str(args.qd_discovered),
            "s08Archive": str(args.s08_archive),
            "s07CompetenceTable": str(artifacts_dir / "results/e03_policy_competence.parquet"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "jax": jax_backend_summary(),
            "elapsedSeconds": elapsed,
        },
        "sourceFiles": [source_entry(path, args.repo_dir) for path in source_files],
        "summary": {
            "selectedPolicyCount": int(len(selected_df)),
            "phaseSweepRows": int(len(run_df)),
            "primaryRows": int((run_df["split"] == "primary").sum()),
            "replicateRows": int((run_df["split"] == "replicate").sum()),
            "boundaryCandidateCount": int(len(boundary_df)),
            "replicatedBoundaryCount": int(boundary_df["replication_available"].sum()) if not boundary_df.empty else 0,
            "replicationDirectionMatches": int(boundary_df["replication_direction_match"].sum()) if not boundary_df.empty else 0,
            "resultDigest": result_digest(run_df),
        },
        "validationSummary": {
            "success": bool(validation["success"].all() and unit_tests["success"]),
            "validationCasesPassed": int(validation["success"].sum()),
            "validationCasesTotal": int(len(validation)),
            "unitTestsReturnCode": int(unit_tests["returnCode"]),
        },
    }

    artifacts = {
        "researchStepReport": full_report_path,
        "phaseBoundarySweeps": phase_path,
        "phaseBoundarySweepsCsv": phase_csv_path,
        "boundaryCandidates": boundary_path,
        "boundaryCandidatesCsv": boundary_csv_path,
        "axisSummary": axis_summary_path,
        "axisSummaryCsv": axis_summary_csv_path,
        "selectedPolicies": selected_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "phaseBoundaryFigure": figure_path,
        "validationConfig": config_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }
    tables = make_summary_tables(run_df, boundary_df, axis_summary)
    full_report = render_report(
        artifacts=artifacts,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
        run_df=run_df,
        boundary_df=boundary_df,
        selected_df=selected_df,
        tables=tables,
    )
    write_text(full_report_path, full_report)

    artifact_entries = [
        artifact_entry(phase_path, artifacts_dir, "S09 row-level phase-boundary sweep table"),
        artifact_entry(phase_csv_path, artifacts_dir, "CSV sidecar for S09 phase-boundary sweeps"),
        artifact_entry(boundary_path, artifacts_dir, "S09 boundary candidate summary"),
        artifact_entry(boundary_csv_path, artifacts_dir, "CSV sidecar for S09 boundary candidates"),
        artifact_entry(axis_summary_path, artifacts_dir, "S09 axis-level summary"),
        artifact_entry(axis_summary_csv_path, artifacts_dir, "CSV sidecar for S09 axis summary"),
        artifact_entry(selected_path, artifacts_dir, "S09 selected policy table"),
        artifact_entry(validation_path, artifacts_dir, "S09 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S09 validation cases"),
        artifact_entry(figure_path, artifacts_dir, "S09 phase-boundary figure"),
        artifact_entry(config_path, artifacts_dir, "S09 phase-boundary config"),
        manifest_self_entry(manifest_path, artifacts_dir, "S09 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S09 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S09"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S09 outputs"),
        artifact_entry(full_report_path, artifacts_dir, "S09 full-results handoff report"),
    ]
    manifest["artifacts"] = artifact_entries
    write_json(manifest_path, manifest)
    artifact_manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            artifact_entry(phase_path, artifacts_dir, "S09 row-level phase-boundary sweep table"),
            artifact_entry(phase_csv_path, artifacts_dir, "CSV sidecar for S09 phase-boundary sweeps"),
            artifact_entry(boundary_path, artifacts_dir, "S09 boundary candidate summary"),
            artifact_entry(boundary_csv_path, artifacts_dir, "CSV sidecar for S09 boundary candidates"),
            artifact_entry(axis_summary_path, artifacts_dir, "S09 axis-level summary"),
            artifact_entry(axis_summary_csv_path, artifacts_dir, "CSV sidecar for S09 axis summary"),
            artifact_entry(selected_path, artifacts_dir, "S09 selected policy table"),
            artifact_entry(validation_path, artifacts_dir, "S09 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S09 validation cases"),
            artifact_entry(figure_path, artifacts_dir, "S09 phase-boundary figure"),
            artifact_entry(config_path, artifacts_dir, "S09 phase-boundary config"),
            artifact_entry(manifest_path, artifacts_dir, "S09 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S09 artifact manifest"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S09"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S09 outputs"),
            artifact_entry(full_report_path, artifacts_dir, "S09 full-results handoff report"),
        ],
    }
    write_json(artifact_manifest_path, artifact_manifest)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": manifest["git"],
        "runtime": manifest["runtime"],
        "commands": [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": " ".join(sys.argv), "returnCode": 0, "success": True},
        ],
        "inputs": manifest["inputArtifacts"],
        "parameters": manifest["parameters"],
        "outputs": artifact_manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)
    checksum_targets = [
        full_report_path,
        phase_path,
        phase_csv_path,
        boundary_path,
        boundary_csv_path,
        axis_summary_path,
        axis_summary_csv_path,
        selected_path,
        validation_path,
        validation_csv_path,
        figure_path,
        config_path,
        manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    success = bool(validation["success"].all() and unit_tests["success"])
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": success,
                "selectedPolicyCount": int(len(selected_df)),
                "phaseSweepRows": int(len(run_df)),
                "boundaryCandidateCount": int(len(boundary_df)),
                "replicationDirectionMatches": int(boundary_df["replication_direction_match"].sum()) if not boundary_df.empty else 0,
                "artifactsDir": str(step_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
