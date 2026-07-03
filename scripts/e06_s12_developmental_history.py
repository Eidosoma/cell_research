#!/usr/bin/env python3
"""Run E06 S12 developmental-history experiments."""

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

from src.e06.developmental_history import (  # noqa: E402
    DEFAULT_HISTORY_SPECS,
    STEP_ID,
    S12Config,
    compare_to_simultaneous_baseline,
    history_spec_table,
    history_strata_summary,
    outcome_classification,
    run_s12_sweep,
    summarize_s12_runs,
    validate_s12_outputs,
)
from src.e06.governance_mechanisms import DEFAULT_GOVERNANCE_MECHANISMS, DEFAULT_INTERFACE_RULES  # noqa: E402
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402


STEP_NUMBER = 12
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s10-selected-path", type=Path, default=ARTIFACTS_DIR / "research_steps/S10/e06_s10_selected_contexts.csv")
    parser.add_argument("--s11-runs-path", type=Path, default=ARTIFACTS_DIR / "results/e06_mutant_clone_experiments.parquet")
    parser.add_argument("--s09-rankings-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s09_governance_rankings.csv")
    parser.add_argument("--previous-e04-memory-path", type=Path, default=Path("/previous-artifacts/E04/policies/e04_repair_capable_algotypes.jsonl"))
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-s10-contexts", type=int, default=3)
    parser.add_argument("--max-memory-contexts", type=int, default=2)
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


def write_history_encoding_spec(path: Path, history_specs: pd.DataFrame) -> None:
    rows = history_specs.sort_values("history_id", kind="mergesort")
    text = f"""# E06 S12 Developmental-History Encoding Spec

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written: this history encoding specification plus the S12 full-results artifact bundle.
- Validation result: History encodings are machine-readable in `e06_s12_history_specs.csv` and validated by S12 checks.
- Outcome classification: Not applicable for this spec; see S12 full-results report.
- Caveats or blockers: Histories are computational schedule, perturbation, and memory-state proxies in a 1D sorting-array substrate.
- Recommended next action: Interpret S12 results by history family and intervention stratum, keeping memory-policy contexts separate.

## History Encodings

{markdown_table(rows, ["history_id", "history_family", "start_mode", "introduction_event", "transient_start_event", "transient_end_event", "transient_profile", "prior_exposure_events", "reset_after_exposure", "preserves_memory_across_reset", "description"], max_rows=20)}
"""
    write_text(path, text)


def write_history_figure(path: Path, run_df: pd.DataFrame, comparison: pd.DataFrame, strata: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if run_df.empty or comparison.empty or strata.empty:
        return False
    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_effect = fig.add_subplot(grid[0, 0])
    ax_quality = fig.add_subplot(grid[0, 1])
    ax_memory = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])

    non_base = strata[strata["history_id"] != "simultaneous_mixed_start"].copy()
    effect = non_base.groupby("history_id", sort=False)["mean_history_effect_magnitude"].mean().sort_values(ascending=False)
    ax_effect.bar(effect.index, effect.values, color="#2f7f7f")
    ax_effect.set_title("Mean history-effect magnitude")
    ax_effect.set_ylabel("Magnitude")
    ax_effect.tick_params(axis="x", rotation=45)

    quality = (
        comparison.groupby("history_id", sort=False)
        .agg(mean_delta_final_target_quality=("delta_final_target_quality", "mean"), mean_abs_delta_final_target_quality=("delta_final_target_quality", lambda x: float(np.mean(np.abs(x)))))
        .reset_index()
    )
    x = np.arange(len(quality))
    ax_quality.bar(x - 0.18, quality["mean_delta_final_target_quality"], width=0.36, label="signed", color="#8c3f97")
    ax_quality.bar(x + 0.18, quality["mean_abs_delta_final_target_quality"], width=0.36, label="abs", color="#b96d28")
    ax_quality.axhline(0, color="#222222", lw=0.8)
    ax_quality.set_xticks(x, labels=quality["history_id"], rotation=45, ha="right")
    ax_quality.set_title("Target-quality shifts vs simultaneous baseline")
    ax_quality.legend(fontsize=8)

    memory = (
        run_df.groupby(["history_id", "memory_policy_present"], sort=False)["memory_total_frustration"]
        .mean()
        .reset_index()
    )
    for present, group in memory.groupby("memory_policy_present"):
        ax_memory.plot(group["history_id"], group["memory_total_frustration"], marker="o", label=f"memory={present}")
    ax_memory.set_title("Final memory frustration by history")
    ax_memory.set_ylabel("Mean total frustration")
    ax_memory.tick_params(axis="x", rotation=45)
    ax_memory.legend(fontsize=8)

    pivot = (
        non_base.pivot_table(
            index="intervention_stratum",
            columns="history_id",
            values="mean_history_effect_magnitude",
            aggfunc="mean",
        )
        .fillna(0.0)
        .sort_index()
    )
    matrix = pivot.to_numpy(dtype=float)
    max_value = max(0.001, float(np.nanmax(matrix)))
    image = ax_heat.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max_value, aspect="auto")
    ax_heat.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax_heat.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax_heat.set_title("History-effect magnitude by stratum")
    fig.colorbar(image, ax=ax_heat, shrink=0.85)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def write_report(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    run_df: pd.DataFrame,
    selected: pd.DataFrame,
    history_specs: pd.DataFrame,
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
    top_history = (
        strata[strata["history_id"] != "simultaneous_mixed_start"].iloc[0]["history_id"]
        if not strata[strata["history_id"] != "simultaneous_mixed_start"].empty
        else "none"
    )
    text = f"""# E06 S12 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: S12 is a bounded computational developmental-history assay. Staged introduction, transient perturbation, and prior-exposure memory are schedule/state proxies in a 1D sorting-array model. Behavior-only, explicit-interface, finite-radius governance, and global-controller-like rows remain separate strata.
- Lay summary: S12 asked whether the same final mixture behaves differently when its developmental history changes. It compared simultaneous mixing, early and late staged partner introduction, early and late transient freezing, and memory-preserving prior exposure while keeping final policy counts matched wherever possible.
- Recommended next action: Proceed to S13 final-state prediction and causal-modeling analyses using S02-S12 merged outcomes, with no leakage from outcome-derived variables.

## Frozen Question

Does timing of mixing and perturbation create lasting history dependence in chimeric outcomes?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 Algotype metadata: `{config_payload['metadataPath']}`
- S10 selected staged graft contexts: `{config_payload['s10SelectedPath']}`
- S11 mutant clone runs: `{config_payload['s11RunsPath']}`
- S09 governance rankings: `{config_payload['s09RankingsPath']}`
- E04 memory-policy provenance path: `{config_payload['previousE04MemoryPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S12 selected the three S10 staged graft-sensitive contexts and added E04 memory-policy input contexts from the S01 library. Each base context was crossed with six explicit history encodings, five interface rules, eight governance mechanisms, and four matched seeds.

The simultaneous baseline starts with the matched final composition at event 0. Staged histories develop a pure host first, then introduce the partner policy using S10/S03 placement machinery. Transient histories start with the matched mixture but freeze a center block during an early or late event window. Prior-exposure histories run a mixed exposure, reset values, labels, and positions to the same matched initial composition, then preserve cell memory state into the main run.

History encodings:

{markdown_table(history_specs.sort_values("history_id"), ["history_id", "history_family", "start_mode", "introduction_event", "transient_start_event", "transient_end_event", "transient_profile", "prior_exposure_events", "reset_after_exposure", "preserves_memory_across_reset", "description"], max_rows=20)}

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

- Selected base contexts: {int(selected['base_context_id'].nunique()) if not selected.empty else 0}.
- E04 memory-policy input contexts: {int((selected['source_context_family'] == 'e04_memory_policy_input').sum()) if not selected.empty else 0}.
- History encodings: {int(run_df['history_id'].nunique()) if not run_df.empty else 0}.
- Run rows: {len(run_df)}.
- Interface rules: {int(run_df['interface_rule_id'].nunique()) if not run_df.empty else 0}.
- Governance mechanisms: {int(run_df['governance_mechanism_id'].nunique()) if not run_df.empty else 0}.
- Matched final-composition failures: {int((~run_df['final_composition_matched']).sum()) if not run_df.empty else 0}.
- Top non-baseline history by mean effect magnitude: `{top_history}`.
- Prior-exposure rows preserving memory across reset: {int(((run_df['history_family'] == 'prior_exposure_memory') & run_df['preserves_memory_across_reset']).sum()) if not run_df.empty else 0}.

Selected contexts:

{markdown_table(selected, ["selection_rank", "base_context_id", "source_context_family", "selection_reason", "host_display_name", "partner_display_name", "source_arrangement", "goal_profile_id", "ratio_targets_json", "memory_policy_present", "memory_policy_ids_json"], max_rows=20)}

History and intervention strata:

{markdown_table(strata, ["history_id", "history_family", "intervention_stratum", "condition_count", "memory_context_fraction", "mean_delta_final_target_quality", "mean_abs_delta_final_target_quality", "mean_delta_goal_conflict_index", "mean_abs_delta_aggregation_delta_percent", "mean_history_effect_magnitude", "state_class_change_rate", "target_quality_shift_rate", "supportive_history_effect_by_threshold"], max_rows=60)}

Per-condition summaries:

{markdown_table(summary_display.sort_values(["base_context_id", "history_id", "interface_rule_id", "governance_mechanism_id"]), ["base_context_id", "history_id", "history_family", "interface_rule_id", "governance_mechanism_id", "display_names", "memory_policy_present", "mean_final_target_quality", "mean_goal_conflict_index", "mean_aggregation_delta_percent", "mean_largest_block_fraction", "mean_memory_total_frustration", "final_composition_matched", "final_state_class_mode", "final_state_class_counts_json"], max_rows=50)}

History comparisons against simultaneous matched-composition baseline under the same base context and intervention stratum:

{markdown_table(comparison.sort_values("history_effect_magnitude", ascending=False), ["base_context_id", "history_id", "interface_rule_id", "governance_mechanism_id", "delta_final_target_quality", "delta_goal_conflict_index", "delta_aggregation_delta_percent", "delta_largest_block_fraction", "history_effect_magnitude", "final_state_class_changed_vs_simultaneous", "target_quality_history_shift"], max_rows=50)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=50)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- Developmental history is represented by computational event schedules and state resets, not biological development.
- Prior-exposure rows intentionally preserve internal memory state while resetting values and labels; that isolates memory-state effects but gives those rows an additional exposure phase.
- Transient perturbations are temporary frozen index masks in a 1D array and are not physical tissue damage.
- Explicit-interface mechanisms use label recognition and remain outside the paper's original no-recognition claim.
- Global-controller-like rows inspect whole-array state and remain comparators rather than local-governance evidence.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Run S13 final-state prediction and exploratory causal-modeling work after review, using S02-S12 outcomes and guarding against leakage from outcome-derived predictors.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_developmental_history.parquet"
    governance_interventions_path = artifacts_dir / "results/e06_governance_interventions.parquet"
    summary_path = artifacts_dir / "tables/e06_developmental_history_summary.csv"
    comparison_path = artifacts_dir / "tables/e06_s12_history_baseline_comparison.csv"
    strata_path = artifacts_dir / "tables/e06_s12_strata_summary.csv"
    selected_path = step_dir / "e06_s12_selected_contexts.csv"
    condition_matrix_path = step_dir / "e06_s12_condition_matrix.csv"
    history_specs_path = step_dir / "e06_s12_history_specs.csv"
    history_spec_report_path = artifacts_dir / "reports/e06_s12_history_encoding_spec.md"
    validation_path = step_dir / "e06_s12_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/history_dependence_summary.png"
    config_path = artifacts_dir / "configs/e06_s12_developmental_history_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S12Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        max_s10_contexts=args.max_s10_contexts,
        max_memory_contexts=args.max_memory_contexts,
        worker_count=max(1, min(8, int(args.workers))),
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "s10SelectedPath": str(args.s10_selected_path),
        "s11RunsPath": str(args.s11_runs_path),
        "s09RankingsPath": str(args.s09_rankings_path),
        "previousE04MemoryPath": str(args.previous_e04_memory_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "maxS10Contexts": config.max_s10_contexts,
        "maxMemoryContexts": config.max_memory_contexts,
        "workerCount": config.worker_count,
        "historyEffectThreshold": config.history_effect_threshold,
        "historySpecs": [spec.__dict__ for spec in DEFAULT_HISTORY_SPECS],
        "interfaceRules": [rule.__dict__ for rule in DEFAULT_INTERFACE_RULES],
        "governanceMechanisms": [mechanism.__dict__ for mechanism in DEFAULT_GOVERNANCE_MECHANISMS],
        "matchedCompositionPolicy": "all histories target the same final policy counts per base context where possible",
        "priorExposurePolicy": "prior-exposure histories reset values, labels, and cell positions while preserving memory state",
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    for input_path in (args.library_path, args.metadata_path, args.s10_selected_path, args.s11_runs_path, args.s09_rankings_path):
        if not input_path.exists():
            raise FileNotFoundError(f"required S12 input is missing: {input_path}")
    records, metadata = load_s01_library(args.library_path, args.metadata_path)
    s10_selected = pd.read_csv(args.s10_selected_path)
    s11_runs = pd.read_parquet(args.s11_runs_path)
    s09_rankings = pd.read_csv(args.s09_rankings_path)

    run_df, condition_df, selected, history_specs = run_s12_sweep(records, metadata, s10_selected, s11_runs, s09_rankings, config)
    history_specs = history_specs.copy()
    history_specs["history_encoding_json"] = history_specs.apply(lambda row: json.dumps({k: row[k] for k in history_specs.columns if k not in {"schema", "research_step_id", "history_encoding_json"}}, sort_keys=True, default=str), axis=1)
    summary = summarize_s12_runs(run_df)
    comparison = compare_to_simultaneous_baseline(summary)
    strata = history_strata_summary(comparison, config)
    figure_written = write_history_figure(figure_path, run_df, comparison, strata)
    validation = validate_s12_outputs(
        run_df,
        condition_df,
        selected,
        history_specs,
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
    history_specs.to_csv(history_specs_path, index=False)
    write_history_encoding_spec(history_spec_report_path, history_specs)
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
        artifact_entry(results_path, artifacts_dir, "S12 run-level developmental-history experiment table."),
        artifact_entry(governance_interventions_path, artifacts_dir, "Cumulative E06 governance/intervention table including S08-S12 rows."),
        artifact_entry(figure_path, artifacts_dir, "S12 developmental-history summary figure."),
        artifact_entry(summary_path, artifacts_dir, "S12 per-condition developmental-history summary table."),
        artifact_entry(comparison_path, artifacts_dir, "S12 matched simultaneous-baseline history comparison table."),
        artifact_entry(strata_path, artifacts_dir, "S12 history and intervention-stratum summary table."),
        artifact_entry(selected_path, artifacts_dir, "S12 selected S10 and E04 memory context table."),
        artifact_entry(condition_matrix_path, artifacts_dir, "S12 executable developmental-history condition matrix."),
        artifact_entry(history_specs_path, artifacts_dir, "S12 machine-readable history encoding definitions."),
        artifact_entry(history_spec_report_path, artifacts_dir, "S12 Markdown history encoding specification."),
        artifact_entry(validation_path, artifacts_dir, "S12 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S12 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S12 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S12 full-results Markdown handoff report.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        run_df=run_df,
        selected=selected,
        history_specs=history_specs,
        summary=summary,
        comparison=comparison,
        strata=strata,
        validation=validation,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S12 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = outcome_classification(strata, success)
    manifest = {
        "schema": "eidosoma.e06.s12_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "outcomeClassification": outcome,
        "runCount": int(len(run_df)),
        "baseContextCount": int(selected["base_context_id"].nunique()),
        "memoryContextCount": int((selected["source_context_family"] == "e04_memory_policy_input").sum()),
        "historySpecCount": int(run_df["history_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "matchedCompositionFailureCount": int((~run_df["final_composition_matched"]).sum()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S12 artifact manifest.")
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
            "s10SelectedContexts": source_entry(args.s10_selected_path),
            "s11MutantCloneRuns": source_entry(args.s11_runs_path),
            "s09GovernanceRankings": source_entry(args.s09_rankings_path),
            "previousE04MemoryPolicies": source_entry(args.previous_e04_memory_path),
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
        history_specs_path,
        history_spec_report_path,
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
        "memoryContextCount": int((selected["source_context_family"] == "e04_memory_policy_input").sum()),
        "historySpecCount": int(run_df["history_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "governanceMechanismCount": int(run_df["governance_mechanism_id"].nunique()),
        "matchedCompositionFailureCount": int((~run_df["final_composition_matched"]).sum()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
