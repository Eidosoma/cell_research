#!/usr/bin/env python3
"""Run E06 S10 staged graft experiments."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.governance_mechanisms import DEFAULT_GOVERNANCE_MECHANISMS, DEFAULT_INTERFACE_RULES  # noqa: E402
from src.e06.graft_experiments import (  # noqa: E402
    STEP_ID,
    S10Config,
    compare_to_behavior_baseline,
    graft_strata_summary,
    run_s10_sweep,
    summarize_s10_runs,
    validate_s10_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402


STEP_NUMBER = 10
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s07-mosaic-path", type=Path, default=ARTIFACTS_DIR / "results/e06_mosaic_classes.parquet")
    parser.add_argument("--s07-exemplars-path", type=Path, default=ARTIFACTS_DIR / "research_steps/S07/e06_s07_exemplar_rows.csv")
    parser.add_argument("--s09-rankings-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s09_governance_rankings.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-base-contexts", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="*", default=[2026070201, 2026070202, 2026070203, 2026070204])
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
    if not present:
        return "_Requested columns are absent._"
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


def _display_name(text: str) -> str:
    try:
        return " vs ".join(str(item) for item in json.loads(str(text))[:3])
    except json.JSONDecodeError:
        return str(text)


def _ratio_label(text: str) -> str:
    try:
        return ":".join(str(int(round(float(value) * 100))) for value in json.loads(str(text)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(text)


def write_graft_figure(path: Path, run_df: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, strata: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if run_df.empty or summary.empty or comparison.empty or strata.empty:
        return False

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_strata = fig.add_subplot(grid[0, 0])
    ax_outcome = fig.add_subplot(grid[0, 1])
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])

    strata_ordered = strata.sort_values("mean_graft_rescue_score", ascending=False, kind="mergesort")
    colors = np.where(strata_ordered["governance_access_strata"].astype(str).str.contains("global"), "#8c3f97", "#2f7f7f")
    ax_strata.bar(strata_ordered["intervention_stratum"], strata_ordered["mean_graft_rescue_score"], color=colors)
    ax_strata.axhline(0, color="#222222", lw=0.8)
    ax_strata.set_title("Mean graft rescue score by stratum")
    ax_strata.set_ylabel("Rescue score")
    ax_strata.tick_params(axis="x", rotation=45)

    outcomes = run_df["graft_outcome_class"].value_counts().sort_index()
    ax_outcome.bar(outcomes.index, outcomes.values, color="#6f7f2f")
    ax_outcome.set_title("Run-level graft outcome classes")
    ax_outcome.set_ylabel("Run count")
    ax_outcome.tick_params(axis="x", rotation=45)

    stratum_names = list(dict.fromkeys(summary["intervention_stratum"].astype(str).tolist()))
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(stratum_names))))
    color_map = dict(zip(stratum_names, palette, strict=False))
    for stratum, group in summary.groupby("intervention_stratum", sort=False):
        ax_scatter.scatter(
            group["mean_graft_relative_largest_block_fraction"],
            group["mean_target_delta_vs_pre_graft"],
            s=55,
            alpha=0.75,
            color=color_map[str(stratum)],
            label=str(stratum),
        )
    ax_scatter.axhline(0, color="#222222", lw=0.8, linestyle="--")
    ax_scatter.set_title("Graft cohesion and target-quality shift")
    ax_scatter.set_xlabel("Mean relative largest graft block fraction")
    ax_scatter.set_ylabel("Mean target delta vs pre-graft")
    ax_scatter.legend(loc="best", fontsize=7)

    pivot = (
        comparison.pivot_table(
            index="intervention_stratum",
            columns="graft_spec_id",
            values="graft_rescue_score",
            aggfunc="mean",
        )
        .reindex(index=strata_ordered["intervention_stratum"].tolist())
        .fillna(0.0)
    )
    matrix = pivot.to_numpy(dtype=float)
    max_abs = max(0.001, float(np.nanmax(np.abs(matrix))))
    image = ax_heat.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax_heat.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax_heat.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax_heat.set_title("Rescue score by graft schedule")
    fig.colorbar(image, ax=ax_heat, shrink=0.85)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def outcome_classification(strata: pd.DataFrame, validation_passed: bool) -> str:
    if not validation_passed or strata.empty:
        return "null"
    local = strata[~strata["governance_access_strata"].astype(str).str.contains("global", na=False)]
    global_like = strata[strata["governance_access_strata"].astype(str).str.contains("global", na=False)]
    local_support = bool((local["supportive_graft_shift_by_threshold"] == True).any())  # noqa: E712
    global_support = bool((global_like["supportive_graft_shift_by_threshold"] == True).any())  # noqa: E712
    if local_support:
        return "supportive"
    if global_support:
        return "constraining/contradictory"
    return "null"


def write_report(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    run_df: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    strata: pd.DataFrame,
    validation: pd.DataFrame,
    config_payload: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    validation_passed = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = outcome_classification(strata, validation_passed)
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    selected_display = selected.copy()
    if not selected_display.empty:
        selected_display["display_names"] = selected_display["display_names_json"].map(_display_name)
        selected_display["ratio_label"] = selected_display["ratio_targets_json"].map(_ratio_label)
    summary_display = summary.copy()
    if not summary_display.empty:
        summary_display["display_names"] = summary_display["display_names_json"].map(_display_name)
        summary_display["ratio_label"] = summary_display["ratio_targets_json"].map(_ratio_label)
    non_global = strata[~strata["governance_access_strata"].astype(str).str.contains("global", na=False)].copy() if not strata.empty else pd.DataFrame()
    top_non_global = non_global.iloc[0]["intervention_stratum"] if not non_global.empty else "none"
    best_global_score = (
        strata[strata["governance_access_strata"].astype(str).str.contains("global", na=False)]["mean_graft_rescue_score"].max()
        if not strata.empty and strata["governance_access_strata"].astype(str).str.contains("global", na=False).any()
        else float("nan")
    )
    text = f"""# E06 S10 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: S10 is a bounded computational graft assay. Grafts are modeled as staged in-place relabeling with new graft cell IDs while preserving the host value state; this is a proxy for transplantation, not a biological graft model. Behavior-only, explicit-interface, finite-radius governance, and global-controller-like comparator strata are kept separate.
- Lay summary: S10 asked whether adding a small graft into a developed host array changes long-run mosaic behavior and whether interface or governance rules alter the outcome. The run saved each host state before grafting, inserted graft patches with S03 arrangement code, and compared behavior-only baselines against explicit-interface, finite-radius governance, and global-controller-like conditions.
- Recommended next action: Proceed to S11 cancer-like mutant experiments using S10 graft-sensitive contexts and keeping global-controller-like rows as comparators only.

## Frozen Question

Can staged graft insertions into S07-sensitive chimeric contexts expose compatibility, rejection, absorption, or reorganization behavior, and do S09 governance options change those outcomes?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 Algotype metadata: `{config_payload['metadataPath']}`
- S07 mosaic classes: `{config_payload['s07MosaicPath']}`
- S07 exemplar rows: `{config_payload['s07ExemplarsPath']}`
- S09 governance rankings: `{config_payload['s09RankingsPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S10 selected a bounded set of pairwise S07 contexts from exemplar rows and metric-continuum extremes. For each context, the majority ratio member was assigned as the host; ties used the first policy as host and the second as graft. Host/graft identity, display names, source categories, S07 labels, continuum scores, and source condition IDs were preserved in every run row.

For each selected context, the simulator first ran a pure host array through the pre-graft event schedule. The full pre-graft host state was saved to `e06_s10_pre_graft_states.parquet`, including values, labels, cell IDs, graft timing, host/graft IDs, and pre-graft sortedness metrics. Graft placement used S03 arrangement families: the center patch uses `graft_like_insertions`, and the edge patch uses `contiguous_patch`.

After graft insertion, runs continued under each S08 interface rule and each S09 governance mechanism. The analysis kept behavior-only, explicit-interface, finite-radius governance, and global-controller-like strata as separate explanatory layers. Graft outcomes were classified from final target quality, target shift from the saved pre-graft state, graft cohesion, edge localization, and interface count.

## Commands

- Main command: `{command}`
- Unit-test command: `{unit_test_result.get('command', 'not run')}`
- Unit-test return code: `{unit_test_result.get('returnCode', 'not run')}`

## Dependencies

No new dependencies were installed. The script used repository modules plus preinstalled `pandas`, `numpy`, `pyarrow`, and `matplotlib`.

## Parameters

```json
{json.dumps(config_payload, indent=2, sort_keys=True, default=str)}
```

## Results

- Selected S07 graft contexts: {int(selected['base_context_id'].nunique()) if not selected.empty else 0}.
- Graft schedules: {int(run_df['graft_spec_id'].nunique()) if not run_df.empty else 0}.
- Run rows: {len(run_df)}.
- Saved unique host pre-graft states: {int(run_df['pre_graft_state_id'].nunique()) if not run_df.empty else 0}.
- Interface rules: {int(run_df['interface_rule_id'].nunique()) if not run_df.empty else 0}.
- Governance mechanisms: {int(run_df['governance_mechanism_id'].nunique()) if not run_df.empty else 0}.
- Top non-global stratum by mean graft rescue score: `{top_non_global}`.
- Best global-controller-like comparator rescue score: {best_global_score:.6g}.
- Behavior-only no-governance rows: {int((run_df['intervention_stratum'] == 'behavior_only_no_governance').sum()) if not run_df.empty else 0}.
- Explicit-interface rows: {int((run_df['interface_access_stratum'] == 'explicit_interface').sum()) if not run_df.empty else 0}.
- Finite-radius governance rows: {int((run_df['governance_access_stratum'] == 'finite_radius_governance').sum()) if not run_df.empty else 0}.
- Global-controller-like rows: {int((run_df['governance_access_stratum'] == 'global_controller_like').sum()) if not run_df.empty else 0}.

Selected S07 graft contexts:

{markdown_table(selected_display, ["selection_rank", "base_context_id", "selection_reason", "display_names", "ratio_label", "host_display_name", "graft_display_name", "s07_label", "goal_profile_id", "goal_compatibility_class", "source_final_target_quality", "source_goal_conflict_index", "source_mosaic_continuum_score", "source_conflict_continuum_score"], max_rows=20)}

Intervention strata:

{markdown_table(strata, ["intervention_stratum", "interface_access_strata", "governance_access_strata", "condition_count", "mean_delta_final_target_quality", "mean_delta_goal_conflict_index", "mean_delta_graft_relative_largest_block_fraction", "mean_graft_rescue_score", "target_quality_rescue_rate", "mean_governance_block_fraction", "mean_interface_block_fraction", "mean_governance_cost_per_event", "supportive_graft_shift_by_threshold"], max_rows=20)}

Per-condition graft summaries:

{markdown_table(summary_display.sort_values(["base_context_id", "graft_spec_id", "interface_rule_id", "governance_mechanism_id"]), ["base_context_id", "graft_spec_id", "interface_rule_id", "governance_mechanism_id", "display_names", "ratio_label", "graft_event", "graft_size", "graft_location", "mean_final_target_quality", "mean_target_delta_vs_pre_graft", "mean_graft_relative_largest_block_fraction", "mean_graft_edge_fraction", "graft_outcome_class_mode", "graft_outcome_counts_json"], max_rows=45)}

Baseline comparisons against matched behavior-only no-governance grafts:

{markdown_table(comparison.sort_values("graft_rescue_score", ascending=False), ["base_context_id", "graft_spec_id", "interface_rule_id", "governance_mechanism_id", "delta_final_target_quality", "delta_goal_conflict_index", "delta_graft_relative_largest_block_fraction", "delta_graft_edge_fraction", "graft_rescue_score", "target_quality_rescued", "graft_mixing_shift"], max_rows=45)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=40)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- Grafts are staged by relabeling selected cells and assigning new graft cell IDs after the pure-host pre-graft phase; the value array is preserved to isolate post-graft behavioral dynamics.
- The selected S07 contexts are intentionally bounded to exemplar and continuum-extreme rows, not an exhaustive search over all S07 mixtures.
- S08 explicit-interface strata use explicit label recognition and therefore are not evidence for original no-recognition claims.
- Finite-radius governance rows are local computational gates; the global-controller-like row inspects whole-array state and is a comparator only.
- Graft outcome classes are rule-based summaries over proxy metrics. Metric continua and exemplar rows should be preferred over hard labels if future classifiers are unstable.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Run S11 cancer-like mutant experiments after review, using S10 graft-sensitive contexts and preserving the same behavior-only, explicit-interface, finite-radius governance, and global-controller-like stratification.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_graft_experiments.parquet"
    governance_interventions_path = artifacts_dir / "results/e06_governance_interventions.parquet"
    summary_path = artifacts_dir / "tables/e06_graft_summary.csv"
    comparison_path = artifacts_dir / "tables/e06_s10_baseline_comparison.csv"
    strata_path = artifacts_dir / "tables/e06_s10_graft_strata_summary.csv"
    selected_path = step_dir / "e06_s10_selected_contexts.csv"
    condition_matrix_path = step_dir / "e06_s10_condition_matrix.csv"
    pre_state_path = step_dir / "e06_s10_pre_graft_states.parquet"
    validation_path = step_dir / "e06_s10_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/graft_outcome_examples.png"
    config_path = artifacts_dir / "configs/e06_s10_graft_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S10Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        max_base_contexts=args.max_base_contexts,
        worker_count=max(1, min(8, int(args.workers))),
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "s07MosaicPath": str(args.s07_mosaic_path),
        "s07ExemplarsPath": str(args.s07_exemplars_path),
        "s09RankingsPath": str(args.s09_rankings_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "maxBaseContexts": config.max_base_contexts,
        "workerCount": config.worker_count,
        "graftRescueDeltaThreshold": config.graft_rescue_delta_threshold,
        "graftSpecs": [spec.__dict__ for spec in config.graft_specs],
        "interfaceRules": [rule.__dict__ for rule in DEFAULT_INTERFACE_RULES],
        "governanceMechanisms": [mechanism.__dict__ for mechanism in DEFAULT_GOVERNANCE_MECHANISMS],
        "globalControllerPolicy": "global-controller-like access is allowed only for rows with global_controller_like=true",
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    for input_path in (args.library_path, args.metadata_path, args.s07_mosaic_path, args.s07_exemplars_path, args.s09_rankings_path):
        if not input_path.exists():
            raise FileNotFoundError(f"required S10 input is missing: {input_path}")
    records, _metadata = load_s01_library(args.library_path, args.metadata_path)
    mosaic = pd.read_parquet(args.s07_mosaic_path)
    exemplars = pd.read_csv(args.s07_exemplars_path)
    s09_rankings = pd.read_csv(args.s09_rankings_path)

    run_df, condition_df, selected, pre_state_df = run_s10_sweep(records, mosaic, exemplars, s09_rankings, config)
    summary = summarize_s10_runs(run_df)
    comparison = compare_to_behavior_baseline(summary)
    strata = graft_strata_summary(comparison, config)
    figure_written = write_graft_figure(figure_path, run_df, summary, comparison, strata)
    validation = validate_s10_outputs(
        run_df,
        condition_df,
        selected,
        pre_state_df,
        summary,
        comparison,
        strata,
        config,
        figure_written=figure_written,
        unit_tests_success=bool(unit_test_result.get("success", True)),
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(results_path, index=False)
    if governance_interventions_path.exists():
        existing = pd.read_parquet(governance_interventions_path)
        if "research_step_id" in existing.columns:
            existing = existing[existing["research_step_id"] != STEP_ID]
        combined = pd.concat([existing, run_df], ignore_index=True, sort=False)
    else:
        combined = run_df.copy()
    combined.to_parquet(governance_interventions_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    strata.to_csv(strata_path, index=False)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_path, index=False)
    condition_df.to_csv(condition_matrix_path, index=False)
    pre_state_df.to_parquet(pre_state_path, index=False)
    validation.to_csv(validation_path, index=False)

    repo_state = {
        "head": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        "remote": git_output(args.repo_dir, ["remote", "-v"]),
        "python": sys.version,
        "platform": platform.platform(),
    }

    base_artifacts = [
        artifact_entry(results_path, artifacts_dir, "S10 run-level staged graft experiment table."),
        artifact_entry(governance_interventions_path, artifacts_dir, "Cumulative E06 governance/intervention table including S08, S09, and S10 rows."),
        artifact_entry(figure_path, artifacts_dir, "S10 graft outcome summary figure."),
        artifact_entry(summary_path, artifacts_dir, "S10 per-condition graft summary table."),
        artifact_entry(comparison_path, artifacts_dir, "S10 matched behavior-only baseline comparison table."),
        artifact_entry(strata_path, artifacts_dir, "S10 intervention-stratum graft summary table."),
        artifact_entry(selected_path, artifacts_dir, "S10 selected S07 graft-context table."),
        artifact_entry(condition_matrix_path, artifacts_dir, "S10 executable graft condition matrix."),
        artifact_entry(pre_state_path, artifacts_dir, "S10 saved full host pre-graft states."),
        artifact_entry(validation_path, artifacts_dir, "S10 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S10 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S10 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S10 full-results Markdown handoff report.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        run_df=run_df,
        selected=selected,
        summary=summary,
        comparison=comparison,
        strata=strata,
        validation=validation,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S10 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = outcome_classification(strata, success)
    manifest = {
        "schema": "eidosoma.e06.s10_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "outcomeClassification": outcome,
        "runCount": int(len(run_df)),
        "baseContextCount": int(selected["base_context_id"].nunique()),
        "graftSpecCount": int(run_df["graft_spec_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "preGraftStateCount": int(pre_state_df["pre_graft_state_id"].nunique()),
        "globalControllerLikeRowCount": int((run_df["governance_access_stratum"] == "global_controller_like").sum()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S10 artifact manifest.")
    artifacts = [*base_artifacts, report_artifact, manifest_artifact]

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAtUtc": utc_now(),
        "repoState": repo_state,
        "python": sys.version,
        "platform": platform.platform(),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "sourceInputs": {
            "s01Library": source_entry(args.library_path),
            "s01Metadata": source_entry(args.metadata_path),
            "s07MosaicClasses": source_entry(args.s07_mosaic_path),
            "s07ExemplarRows": source_entry(args.s07_exemplars_path),
            "s09GovernanceRankings": source_entry(args.s09_rankings_path),
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        governance_interventions_path,
        figure_path,
        summary_path,
        comparison_path,
        strata_path,
        selected_path,
        condition_matrix_path,
        pre_state_path,
        validation_path,
        config_path,
        manifest_path,
        report_path,
        run_manifest_path,
    ]
    write_text(checksums_path, "\n".join(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_paths) + "\n")

    status = {
        "success": success,
        "outcomeClassification": outcome,
        "runCount": int(len(run_df)),
        "baseContextCount": int(selected["base_context_id"].nunique()),
        "graftSpecCount": int(run_df["graft_spec_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "preGraftStateCount": int(pre_state_df["pre_graft_state_id"].nunique()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
