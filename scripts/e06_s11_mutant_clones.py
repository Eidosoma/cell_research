#!/usr/bin/env python3
"""Run E06 S11 cancer-like mutant clone experiments."""

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
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402
from src.e06.mutant_clones import (  # noqa: E402
    DEFAULT_SELFISH_OBJECTIVES,
    STEP_ID,
    S11Config,
    compare_to_behavior_baseline,
    mutant_strata_summary,
    run_s11_sweep,
    summarize_s11_runs,
    validate_s11_outputs,
)


STEP_NUMBER = 11
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s10-runs-path", type=Path, default=ARTIFACTS_DIR / "results/e06_graft_experiments.parquet")
    parser.add_argument("--s10-selected-path", type=Path, default=ARTIFACTS_DIR / "research_steps/S10/e06_s10_selected_contexts.csv")
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


def write_selfish_objective_spec(path: Path, objective_df: pd.DataFrame) -> None:
    rows = objective_df.sort_values("objective_id", kind="mergesort")
    text = f"""# E06 S11 Computational Selfish Objective Spec

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written: this objective specification plus the S11 full-results artifact bundle.
- Validation result: Objectives are machine-readable in `e06_s11_selfish_objectives.csv` and validated by S11 checks.
- Outcome classification: Not applicable for this spec; see S11 full-results report.
- Caveats or blockers: These are computational selfish-objective proxies only and do not establish biological cancer, cell division, adhesion, or migration mechanisms.
- Recommended next action: Interpret S11 clone outcomes by objective family and intervention stratum.

## Objective Definitions

{markdown_table(rows, ["objective_id", "objective_family", "score_name", "allows_replication", "definition", "action_rule", "biological_analogy_caveat"], max_rows=20)}
"""
    write_text(path, text)


def write_mutant_figure(path: Path, run_df: pd.DataFrame, comparison: pd.DataFrame, strata: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if run_df.empty or comparison.empty or strata.empty:
        return False

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_strata = fig.add_subplot(grid[0, 0])
    ax_outcome = fig.add_subplot(grid[0, 1])
    ax_objective = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])

    strata_ordered = strata.sort_values("mean_containment_effect_score", ascending=False, kind="mergesort")
    colors = np.where(strata_ordered["governance_access_strata"].astype(str).str.contains("global"), "#8c3f97", "#2f7f7f")
    ax_strata.bar(strata_ordered["intervention_stratum"], strata_ordered["mean_containment_effect_score"], color=colors)
    ax_strata.axhline(0, color="#222222", lw=0.8)
    ax_strata.set_title("Mean containment-effect score by stratum")
    ax_strata.set_ylabel("Containment-effect score")
    ax_strata.tick_params(axis="x", rotation=45)

    outcomes = run_df["mutant_outcome_class"].value_counts().sort_index()
    ax_outcome.bar(outcomes.index, outcomes.values, color="#6f7f2f")
    ax_outcome.set_title("Run-level mutant outcome classes")
    ax_outcome.set_ylabel("Run count")
    ax_outcome.tick_params(axis="x", rotation=45)

    objective_summary = run_df.groupby("selfish_objective_id", sort=False).agg(
        mean_final_clone_fraction=("final_clone_fraction", "mean"),
        takeover_rate=("takeover_success", "mean"),
        damage_rate=("damaging_clone", "mean"),
    )
    x = np.arange(len(objective_summary))
    width = 0.25
    ax_objective.bar(x - width, objective_summary["mean_final_clone_fraction"], width=width, label="clone fraction", color="#2f7f7f")
    ax_objective.bar(x, objective_summary["takeover_rate"], width=width, label="takeover rate", color="#8c3f97")
    ax_objective.bar(x + width, objective_summary["damage_rate"], width=width, label="damage rate", color="#b96d28")
    ax_objective.set_xticks(x, labels=objective_summary.index, rotation=30, ha="right")
    ax_objective.set_title("Mutant behavior by selfish objective")
    ax_objective.legend(loc="best", fontsize=8)

    pivot = (
        comparison.pivot_table(
            index="intervention_stratum",
            columns="selfish_objective_id",
            values="containment_effect_score",
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
    ax_heat.set_title("Containment score by objective and stratum")
    fig.colorbar(image, ax=ax_heat, shrink=0.85)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def outcome_classification(strata: pd.DataFrame, validation_passed: bool) -> str:
    if not validation_passed or strata.empty:
        return "null"
    local = strata[~strata["governance_access_strata"].astype(str).str.contains("global", na=False)]
    global_like = strata[strata["governance_access_strata"].astype(str).str.contains("global", na=False)]
    local_support = bool((local["supportive_containment_by_threshold"] == True).any())  # noqa: E712
    global_support = bool((global_like["supportive_containment_by_threshold"] == True).any())  # noqa: E712
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
    objective_df: pd.DataFrame,
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
    summary_display = summary.copy()
    if not summary_display.empty:
        summary_display["display_names"] = summary_display["display_names_json"].map(_display_name)
    non_global = strata[~strata["governance_access_strata"].astype(str).str.contains("global", na=False)].copy() if not strata.empty else pd.DataFrame()
    top_non_global = non_global.iloc[0]["intervention_stratum"] if not non_global.empty else "none"
    best_global_score = (
        strata[strata["governance_access_strata"].astype(str).str.contains("global", na=False)]["mean_containment_effect_score"].max()
        if not strata.empty and strata["governance_access_strata"].astype(str).str.contains("global", na=False).any()
        else float("nan")
    )
    text = f"""# E06 S11 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: S11 is a bounded computational mutant-clone assay. "Cancer-like" means explicitly defined selfish clone objectives in a sorting-array proxy; it is not biological cancer validation. Behavior-only, explicit-interface, finite-radius governance, and global-controller-like rows remain separate strata.
- Lay summary: S11 introduced small mutant clones into S10 graft-sensitive contexts and asked whether selfish objectives lead to containment, local invasion, or takeover. Clone size and location were logged, objectives were defined in machine-readable form, and every intervention was compared to matched behavior-only no-governance baselines.
- Recommended next action: Proceed to S12 developmental-history experiments using S10/S11 staged-introduction machinery and preserving the same intervention strata.

## Frozen Question

Can small selfish clones be contained by local collectives, or do they take over and degrade global target quality?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 Algotype metadata: `{config_payload['metadataPath']}`
- S10 graft runs: `{config_payload['s10RunsPath']}`
- S10 selected graft contexts: `{config_payload['s10SelectedPath']}`
- S09 governance rankings: `{config_payload['s09RankingsPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S11 selected the S10 graft-sensitive base contexts and ranked them by S10 outcome diversity, target-quality displacement after grafting, and graft-block cohesion. The mutant parent identity is the S10 graft identity, but the S11 mutant label is a new computational proxy whose behavior is governed by one of three explicit selfish objectives.

The run first developed a pure host array, then introduced a small clone using the S03/S10 placement families. The two clone schedules were a 5-cell early center clone and a 10-cell late left-edge clone. Each run logs clone introduction event, initial size, location, placement reference, initial indices, and initial clone state.

The three computational selfish objectives were:

{markdown_table(objective_df.sort_values("objective_id"), ["objective_id", "objective_family", "score_name", "allows_replication", "definition", "action_rule", "biological_analogy_caveat"], max_rows=10)}

After clone introduction, host cells continued using their source S01 policy while mutant cells used the configured selfish objective. Swap-like proposals passed through the S08 interface gate and S09 governance gate. The `replicative_takeover` objective can convert adjacent host identity labels to mutant labels after those gates accept; values are not duplicated, so the assay measures identity takeover rather than biological cell division.

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

- Selected S10 graft-sensitive contexts: {int(selected['base_context_id'].nunique()) if not selected.empty else 0}.
- Clone schedules: {int(run_df['clone_spec_id'].nunique()) if not run_df.empty else 0}.
- Selfish objectives: {int(run_df['selfish_objective_id'].nunique()) if not run_df.empty else 0}.
- Run rows: {len(run_df)}.
- Saved unique initial clone states: {int(run_df['initial_clone_state_id'].nunique()) if not run_df.empty else 0}.
- Interface rules: {int(run_df['interface_rule_id'].nunique()) if not run_df.empty else 0}.
- Governance mechanisms: {int(run_df['governance_mechanism_id'].nunique()) if not run_df.empty else 0}.
- Top non-global stratum by mean containment-effect score: `{top_non_global}`.
- Best global-controller-like comparator containment-effect score: {best_global_score:.6g}.
- Behavior-only no-governance rows: {int((run_df['intervention_stratum'] == 'behavior_only_no_governance').sum()) if not run_df.empty else 0}.
- Explicit-interface rows: {int((run_df['interface_access_stratum'] == 'explicit_interface').sum()) if not run_df.empty else 0}.
- Finite-radius governance rows: {int((run_df['governance_access_stratum'] == 'finite_radius_governance').sum()) if not run_df.empty else 0}.
- Global-controller-like rows: {int((run_df['governance_access_stratum'] == 'global_controller_like').sum()) if not run_df.empty else 0}.

Selected S10 contexts:

{markdown_table(selected, ["selection_rank", "base_context_id", "s11_selection_reason", "host_display_name", "graft_display_name", "s07_label", "goal_profile_id", "s10_graft_outcome_classes", "s10_mean_target_delta_vs_pre_graft", "s10_max_abs_target_delta_vs_pre_graft", "s10_graft_sensitivity_score"], max_rows=20)}

Intervention strata:

{markdown_table(strata, ["intervention_stratum", "interface_access_strata", "governance_access_strata", "condition_count", "mean_delta_final_clone_fraction", "mean_delta_target_quality_damage", "mean_delta_takeover_rate", "mean_delta_containment_success_rate", "mean_containment_effect_score", "takeover_reduction_rate", "damage_reduction_rate", "mean_governance_block_fraction", "mean_interface_block_fraction", "mean_governance_cost_per_event", "supportive_containment_by_threshold"], max_rows=20)}

Per-condition mutant summaries:

{markdown_table(summary_display.sort_values(["base_context_id", "clone_spec_id", "selfish_objective_id", "interface_rule_id", "governance_mechanism_id"]), ["base_context_id", "clone_spec_id", "selfish_objective_id", "interface_rule_id", "governance_mechanism_id", "display_names", "clone_initial_size", "clone_initial_location", "mean_final_clone_fraction", "mean_clone_growth_delta", "mean_target_quality_damage", "takeover_rate", "containment_success_rate", "damaging_clone_rate", "mutant_outcome_class_mode", "mutant_outcome_counts_json"], max_rows=50)}

Baseline comparisons against matched behavior-only no-governance clone runs:

{markdown_table(comparison.sort_values("containment_effect_score", ascending=False), ["base_context_id", "clone_spec_id", "selfish_objective_id", "interface_rule_id", "governance_mechanism_id", "delta_final_clone_fraction", "delta_target_quality_damage", "delta_takeover_rate", "delta_containment_success_rate", "containment_effect_score", "takeover_reduced", "damage_reduced"], max_rows=50)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=50)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- "Cancer-like" is used only as a computational analogy for selfish clone objectives.
- The replication objective converts identity labels without duplicating values, biomass, or physical cells.
- Explicit-interface rules use label recognition and remain outside the paper's original no-recognition claim.
- Finite-radius governance rows are local computational gates; global-controller-like rows inspect whole-array state and remain comparators only.
- S11 is bounded to S10 graft-sensitive contexts, not a full mutant-clone phase atlas over all E06 mixtures.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Run S12 developmental-history experiments after review, reusing staged-introduction machinery from S10 and S11 while preserving intervention-stratum separation.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_mutant_clone_experiments.parquet"
    governance_interventions_path = artifacts_dir / "results/e06_governance_interventions.parquet"
    summary_path = artifacts_dir / "tables/e06_mutant_clone_summary.csv"
    comparison_path = artifacts_dir / "tables/e06_s11_baseline_comparison.csv"
    strata_path = artifacts_dir / "tables/e06_s11_strata_summary.csv"
    selected_path = step_dir / "e06_s11_selected_contexts.csv"
    condition_matrix_path = step_dir / "e06_s11_condition_matrix.csv"
    initial_state_path = step_dir / "e06_s11_initial_clone_states.parquet"
    objective_path = step_dir / "e06_s11_selfish_objectives.csv"
    objective_spec_path = artifacts_dir / "reports/e06_s11_selfish_objective_spec.md"
    validation_path = step_dir / "e06_s11_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/mutant_takeover_curves.png"
    config_path = artifacts_dir / "configs/e06_s11_mutant_clone_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S11Config(
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
        "s10RunsPath": str(args.s10_runs_path),
        "s10SelectedPath": str(args.s10_selected_path),
        "s09RankingsPath": str(args.s09_rankings_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "maxBaseContexts": config.max_base_contexts,
        "workerCount": config.worker_count,
        "containmentEffectThreshold": config.containment_effect_threshold,
        "cloneSpecs": [spec.__dict__ for spec in config.clone_specs],
        "selfishObjectives": [objective.__dict__ for objective in DEFAULT_SELFISH_OBJECTIVES],
        "interfaceRules": [rule.__dict__ for rule in DEFAULT_INTERFACE_RULES],
        "governanceMechanisms": [mechanism.__dict__ for mechanism in DEFAULT_GOVERNANCE_MECHANISMS],
        "globalControllerPolicy": "global-controller-like access is allowed only for rows with global_controller_like=true",
        "computationalAnalogyPolicy": "cancer-like means selfish clone objective in a sorting-array proxy, not biological validation",
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    for input_path in (args.library_path, args.metadata_path, args.s10_runs_path, args.s10_selected_path, args.s09_rankings_path):
        if not input_path.exists():
            raise FileNotFoundError(f"required S11 input is missing: {input_path}")
    records, _metadata = load_s01_library(args.library_path, args.metadata_path)
    s10_runs = pd.read_parquet(args.s10_runs_path)
    s10_selected = pd.read_csv(args.s10_selected_path)
    s09_rankings = pd.read_csv(args.s09_rankings_path)

    run_df, condition_df, selected, initial_state_df, objectives = run_s11_sweep(records, s10_selected, s10_runs, s09_rankings, config)
    summary = summarize_s11_runs(run_df)
    comparison = compare_to_behavior_baseline(summary)
    strata = mutant_strata_summary(comparison, config)
    figure_written = write_mutant_figure(figure_path, run_df, comparison, strata)
    validation = validate_s11_outputs(
        run_df,
        condition_df,
        selected,
        initial_state_df,
        objectives,
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
    initial_state_df.to_parquet(initial_state_path, index=False)
    objectives.to_csv(objective_path, index=False)
    write_selfish_objective_spec(objective_spec_path, objectives)
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
        artifact_entry(results_path, artifacts_dir, "S11 run-level mutant clone experiment table."),
        artifact_entry(governance_interventions_path, artifacts_dir, "Cumulative E06 governance/intervention table including S08-S11 rows."),
        artifact_entry(figure_path, artifacts_dir, "S11 mutant takeover/containment summary figure."),
        artifact_entry(summary_path, artifacts_dir, "S11 per-condition mutant clone summary table."),
        artifact_entry(comparison_path, artifacts_dir, "S11 matched behavior-only baseline comparison table."),
        artifact_entry(strata_path, artifacts_dir, "S11 intervention-stratum mutant containment summary table."),
        artifact_entry(selected_path, artifacts_dir, "S11 selected S10 graft-sensitive context table."),
        artifact_entry(condition_matrix_path, artifacts_dir, "S11 executable mutant clone condition matrix."),
        artifact_entry(initial_state_path, artifacts_dir, "S11 saved full initial clone states."),
        artifact_entry(objective_path, artifacts_dir, "S11 machine-readable selfish objective definitions."),
        artifact_entry(objective_spec_path, artifacts_dir, "S11 Markdown selfish objective specification."),
        artifact_entry(validation_path, artifacts_dir, "S11 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S11 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S11 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S11 full-results Markdown handoff report.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        run_df=run_df,
        selected=selected,
        objective_df=objectives,
        summary=summary,
        comparison=comparison,
        strata=strata,
        validation=validation,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S11 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = outcome_classification(strata, success)
    manifest = {
        "schema": "eidosoma.e06.s11_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "outcomeClassification": outcome,
        "runCount": int(len(run_df)),
        "baseContextCount": int(selected["base_context_id"].nunique()),
        "cloneSpecCount": int(run_df["clone_spec_id"].nunique()),
        "selfishObjectiveCount": int(run_df["selfish_objective_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "initialCloneStateCount": int(initial_state_df["initial_clone_state_id"].nunique()),
        "globalControllerLikeRowCount": int((run_df["governance_access_stratum"] == "global_controller_like").sum()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S11 artifact manifest.")
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
            "s10GraftRuns": source_entry(args.s10_runs_path),
            "s10SelectedContexts": source_entry(args.s10_selected_path),
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
        initial_state_path,
        objective_path,
        objective_spec_path,
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
        "cloneSpecCount": int(run_df["clone_spec_id"].nunique()),
        "selfishObjectiveCount": int(run_df["selfish_objective_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "initialCloneStateCount": int(initial_state_df["initial_clone_state_id"].nunique()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
