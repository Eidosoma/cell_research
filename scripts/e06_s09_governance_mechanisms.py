#!/usr/bin/env python3
"""Run E06 S09 governance-mechanism intervention sweep."""

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

from src.e06.governance_mechanisms import (  # noqa: E402
    DEFAULT_GOVERNANCE_MECHANISMS,
    DEFAULT_INTERFACE_RULES,
    STEP_ID,
    S09Config,
    compare_to_no_governance_baseline,
    governance_rankings,
    information_access_strata,
    run_s09_sweep,
    summarize_s09_runs,
    validate_s09_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402


STEP_NUMBER = 9
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s04-goal-path", type=Path, default=ARTIFACTS_DIR / "results/e06_goal_compatibility.parquet")
    parser.add_argument("--s08-strata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s08_mechanism_strata.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-base-conditions", type=int, default=8)
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
    display = frame[columns].head(max_rows).copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.to_dict(orient="records"):
        values = []
        for column in columns:
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
        lines.append("| ... | " + f"{len(frame) - max_rows} more rows omitted from report table" + " |" * (len(columns) - 1))
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


def write_governance_figure(path: Path, comparison: pd.DataFrame, rankings: pd.DataFrame, info_strata: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if comparison.empty or rankings.empty:
        return False
    ordered = rankings.sort_values("mean_rescue_effect_score", ascending=False, kind="mergesort").copy()
    colors = np.where(ordered["global_controller_like"].astype(bool), "#8c3f97", "#2f7f7f")
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_score = fig.add_subplot(grid[0, 0])
    ax_quality = fig.add_subplot(grid[0, 1])
    ax_cost = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])

    ax_score.bar(ordered["governance_mechanism_id"], ordered["mean_rescue_effect_score"], color=colors)
    ax_score.axhline(0, color="#222222", lw=0.8)
    ax_score.set_title("Mean rescue-effect score vs no governance")
    ax_score.set_ylabel("Rescue score")
    ax_score.tick_params(axis="x", rotation=45)

    ax_quality.bar(ordered["governance_mechanism_id"], ordered["mean_delta_final_target_quality"], color=colors)
    ax_quality.axhline(0, color="#222222", lw=0.8)
    ax_quality.set_title("Mean target-quality delta")
    ax_quality.set_ylabel("Delta assigned-goal quality")
    ax_quality.tick_params(axis="x", rotation=45)

    local = comparison[comparison["governance_mechanism_id"] != "no_governance"].copy()
    for global_like, group in local.groupby("global_controller_like", sort=False):
        ax_cost.scatter(
            group["mean_governance_cost_per_event"],
            group["rescue_effect_score"],
            s=50,
            alpha=0.7,
            label="global-controller-like" if global_like else "finite-radius local",
        )
    ax_cost.axhline(0, color="#222222", lw=0.8, linestyle="--")
    ax_cost.set_title("Cost and rescue by condition")
    ax_cost.set_xlabel("Governance cost units per event")
    ax_cost.set_ylabel("Rescue score")
    ax_cost.legend(loc="best", fontsize=8)

    pivot = (
        comparison.pivot_table(
            index="governance_mechanism_id",
            columns="interface_rule_id",
            values="rescue_effect_score",
            aggfunc="mean",
        )
        .reindex(index=ordered["governance_mechanism_id"].tolist())
        .fillna(0.0)
    )
    matrix = pivot.to_numpy(dtype=float)
    max_abs = max(0.001, float(np.nanmax(np.abs(matrix))))
    image = ax_heat.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax_heat.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax_heat.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax_heat.set_title("Rescue score by interface stratum")
    fig.colorbar(image, ax=ax_heat, shrink=0.85)

    if not info_strata.empty:
        global_note = info_strata[info_strata["global_controller_like"]]
        note = f"Global-controller-like strata rows: {len(global_note)}"
        ax_cost.text(0.02, 0.02, note, transform=ax_cost.transAxes, ha="left", va="bottom", fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def outcome_classification(rankings: pd.DataFrame, validation_passed: bool) -> str:
    if not validation_passed or rankings.empty:
        return "null"
    local = rankings[(rankings["governance_mechanism_id"] != "no_governance") & (~rankings["global_controller_like"].astype(bool))]
    global_like = rankings[rankings["global_controller_like"].astype(bool)]
    local_support = bool((local["supportive_rescue_by_threshold"] == True).any())  # noqa: E712
    global_support = bool((global_like["supportive_rescue_by_threshold"] == True).any())  # noqa: E712
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
    rankings: pd.DataFrame,
    info_strata: pd.DataFrame,
    validation: pd.DataFrame,
    config_payload: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    validation_passed = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = outcome_classification(rankings, validation_passed)
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    selected_display = selected.copy()
    if not selected_display.empty:
        selected_display["display_names"] = selected_display["display_names_json"].map(_display_name)
        selected_display["ratio_label"] = selected_display["ratio_targets_json"].map(_ratio_label)
    summary_display = summary.copy()
    if not summary_display.empty:
        summary_display["display_names"] = summary_display["display_names_json"].map(_display_name)
        summary_display["ratio_label"] = summary_display["ratio_targets_json"].map(_ratio_label)
    local_rankings = rankings[~rankings["global_controller_like"].astype(bool)].copy() if not rankings.empty else pd.DataFrame()
    top_local = local_rankings.iloc[0]["governance_mechanism_id"] if not local_rankings.empty else "none"
    best_global_score = (
        rankings[rankings["global_controller_like"].astype(bool)]["mean_rescue_effect_score"].max()
        if not rankings.empty and rankings["global_controller_like"].astype(bool).any()
        else float("nan")
    )
    text = f"""# E06 S09 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: S09 is bounded to selected S04 conflict cases and uses computational governance gates, not biological governance mechanisms. Behavior-only and explicit-interface S08 strata are kept separate. The `global_reference_controller` is intentionally labeled global-controller-like and should be treated only as a comparator, not as local governance evidence.
- Lay summary: S09 asked whether simple local governance rules can improve conflicted chimeric sorting collectives. The run replayed S04 conflict cases under each S08 interface stratum, then compared no governance, finite-radius local governance mechanisms, and one explicitly labeled global comparator. Results are ranked by target-quality rescue, goal-conflict shift, and governance cost.
- Recommended next action: Proceed to S10 graft experiments using finite-radius S09 mechanisms as local-governance candidates and keeping any global-controller-like comparator separate.

## Frozen Question

Can local governance mechanisms rescue conflict or steer mixed collectives without a full global controller?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 Algotype metadata: `{config_payload['metadataPath']}`
- S04 goal-compatibility runs: `{config_payload['s04GoalPath']}`
- S08 mechanism strata: `{config_payload['s08StrataPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S09 selected a bounded panel of S04 conflict cases from `opposite_goal`, `partially_compatible`, and `unrelated_goal` profiles. Selection prioritized high goal-alignment gaps, reference-target loss, aggregation shifts, and conflict/tension goal-state modes while preserving class diversity where possible.

Each selected S04 conflict case was replayed with matched seeds across the five S08 interface strata: one behavior-only no-recognition baseline and four explicit-interface intervention strata. Within each base/interface stratum, `no_governance` was the matched baseline and all governance mechanisms were compared against it.

Governance mechanisms were implemented as gates on valid local swap proposals before the S08 interface gate. Finite-radius mechanisms used only local labels, local value/goal proxies, fixed sparse leader/pacemaker/organizer positions, or local quorum statistics. The global comparator inspected whole-array increasing sortedness and is explicitly labeled `global_controller_like=true`.

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

- Selected S04 base conflict cases: {int(selected['base_condition_id'].nunique()) if not selected.empty else 0}.
- Governance run rows: {len(run_df)}.
- Interface strata: {int(run_df['interface_rule_id'].nunique()) if not run_df.empty else 0}.
- Governance mechanisms: {int(run_df['governance_mechanism_id'].nunique()) if not run_df.empty else 0}.
- Top finite-radius local mechanism by mean rescue score: `{top_local}`.
- Best global-controller-like comparator rescue score: {best_global_score:.6g}.
- Behavior-only rows: {int((run_df['interface_rule_id'] == 'behavior_only').sum()) if not run_df.empty else 0}; explicit-interface rows: {int((run_df['interface_rule_id'] != 'behavior_only').sum()) if not run_df.empty else 0}.

Selected S04 conflict cases:

{markdown_table(selected_display, ["selection_rank", "base_condition_id", "selection_reason", "display_names", "ratio_label", "arrangement", "goal_profile_id", "goal_compatibility_class", "source_mean_final_target_quality", "source_mean_reference_increasing_score", "source_mean_goal_conflict_index", "source_mean_aggregation_delta_percent"], max_rows=20)}

Governance rankings:

{markdown_table(rankings, ["governance_mechanism_id", "governance_mechanism_family", "global_controller_like", "governance_influence_radius_label", "mean_delta_final_target_quality", "mean_delta_goal_conflict_index", "mean_governance_block_fraction", "mean_governance_cost_per_event", "target_quality_rescue_rate", "conflict_reduction_rate", "mean_rescue_effect_score", "supportive_rescue_by_threshold"], max_rows=20)}

Information-access strata:

{markdown_table(info_strata.sort_values(["global_controller_like", "governance_influence_radius_label", "interface_rule_id"]), ["global_controller_like", "governance_influence_radius_label", "governance_information_access_class", "interface_rule_id", "explicit_recognition_used", "mechanism_count", "mean_delta_final_target_quality", "mean_delta_goal_conflict_index", "mean_rescue_effect_score", "mean_governance_cost_per_event"], max_rows=40)}

Per-condition summaries:

{markdown_table(summary_display.sort_values(["base_condition_id", "interface_rule_id", "governance_mechanism_id"]), ["base_condition_id", "interface_rule_id", "governance_mechanism_id", "display_names", "ratio_label", "goal_profile_id", "mean_final_target_quality", "mean_goal_conflict_index", "mean_governance_block_fraction", "mean_governance_cost_per_event", "mean_interface_block_fraction"], max_rows=40)}

Baseline comparisons against matched no-governance rows:

{markdown_table(comparison[comparison["governance_mechanism_id"] != "no_governance"].sort_values("rescue_effect_score", ascending=False), ["base_condition_id", "interface_rule_id", "governance_mechanism_id", "delta_final_target_quality", "delta_goal_conflict_index", "delta_aggregation_delta_percent", "delta_work_count", "mean_governance_block_fraction", "mean_governance_cost_per_event", "rescue_effect_score"], max_rows=45)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=40)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- Local voting, leaders, pacemakers, quorum signals, conflict-resolution signals, and organizers are computational swap-gate abstractions.
- Fixed sparse leaders, pacemakers, and organizers have finite influence radii but their placement is manually configured rather than evolved or inferred.
- The global comparator deliberately violates the local-governance hypothesis; it is retained only to show how much a controller with whole-array state can do under the same bounded cases.
- The sweep is bounded to eight selected S04 conflict cases and all five S08 interface strata, not an exhaustive chimeric governance atlas.
- S08 explicit-interface strata already introduced explicit label recognition, so S09 reports behavior-only and explicit-interface outcomes separately.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Run S10 graft experiments after review, using S09 finite-radius governance rankings as candidate local interventions while preserving behavior-only, explicit-interface, and global-controller-like strata as separate explanatory layers.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_governance_mechanisms.parquet"
    governance_interventions_path = artifacts_dir / "results/e06_governance_interventions.parquet"
    summary_path = artifacts_dir / "tables/e06_governance_summary.csv"
    comparison_path = artifacts_dir / "tables/e06_s09_baseline_comparison.csv"
    rankings_path = artifacts_dir / "tables/e06_s09_governance_rankings.csv"
    info_strata_path = artifacts_dir / "tables/e06_s09_information_access_strata.csv"
    selected_path = step_dir / "e06_s09_selected_conflict_cases.csv"
    condition_matrix_path = step_dir / "e06_s09_condition_matrix.csv"
    validation_path = step_dir / "e06_s09_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/governance_effects.png"
    config_path = artifacts_dir / "configs/e06_s09_governance_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S09Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        max_base_conditions=args.max_base_conditions,
        worker_count=max(1, min(8, int(args.workers))),
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "s04GoalPath": str(args.s04_goal_path),
        "s08StrataPath": str(args.s08_strata_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "maxBaseConditions": config.max_base_conditions,
        "workerCount": config.worker_count,
        "rescueDeltaThreshold": config.rescue_delta_threshold,
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

    for path in (args.library_path, args.metadata_path, args.s04_goal_path, args.s08_strata_path):
        if not path.exists():
            raise FileNotFoundError(f"required S09 input is missing: {path}")
    records, _metadata = load_s01_library(args.library_path, args.metadata_path)
    s04_runs = pd.read_parquet(args.s04_goal_path)
    s08_strata = pd.read_csv(args.s08_strata_path)

    run_df, condition_df, selected = run_s09_sweep(records, s04_runs, s08_strata, config)
    summary = summarize_s09_runs(run_df)
    comparison = compare_to_no_governance_baseline(summary)
    rankings = governance_rankings(comparison, config)
    info_strata = information_access_strata(comparison)
    figure_written = write_governance_figure(figure_path, comparison, rankings, info_strata)
    validation = validate_s09_outputs(
        run_df,
        condition_df,
        selected,
        summary,
        comparison,
        rankings,
        info_strata,
        s08_strata,
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
    rankings.to_csv(rankings_path, index=False)
    info_strata.to_csv(info_strata_path, index=False)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_path, index=False)
    condition_df.to_csv(condition_matrix_path, index=False)
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
        artifact_entry(results_path, artifacts_dir, "S09 run-level governance mechanism table."),
        artifact_entry(governance_interventions_path, artifacts_dir, "Cumulative E06 governance/intervention table including S08 and S09 rows."),
        artifact_entry(figure_path, artifacts_dir, "S09 governance effect summary figure."),
        artifact_entry(summary_path, artifacts_dir, "S09 per-condition governance summary table."),
        artifact_entry(comparison_path, artifacts_dir, "S09 matched no-governance baseline comparison table."),
        artifact_entry(rankings_path, artifacts_dir, "S09 governance mechanism rescue and cost ranking table."),
        artifact_entry(info_strata_path, artifacts_dir, "S09 information-access and global-controller stratification table."),
        artifact_entry(selected_path, artifacts_dir, "S09 selected S04 conflict-case table."),
        artifact_entry(condition_matrix_path, artifacts_dir, "S09 executable condition matrix."),
        artifact_entry(validation_path, artifacts_dir, "S09 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S09 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S09 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S09 full-results Markdown handoff report.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        run_df=run_df,
        selected=selected,
        summary=summary,
        comparison=comparison,
        rankings=rankings,
        info_strata=info_strata,
        validation=validation,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S09 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = outcome_classification(rankings, success)
    manifest = {
        "schema": "eidosoma.e06.s09_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "outcomeClassification": outcome,
        "runCount": int(len(run_df)),
        "baseConditionCount": int(selected["base_condition_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "globalControllerLikeMechanismCount": int(run_df[run_df["global_controller_like"]]["governance_mechanism_id"].nunique()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S09 artifact manifest.")
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
            "s04GoalCompatibility": source_entry(args.s04_goal_path),
            "s08MechanismStrata": source_entry(args.s08_strata_path),
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        governance_interventions_path,
        figure_path,
        summary_path,
        comparison_path,
        rankings_path,
        info_strata_path,
        selected_path,
        condition_matrix_path,
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
        "baseConditionCount": int(selected["base_condition_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
