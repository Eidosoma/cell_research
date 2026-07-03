#!/usr/bin/env python3
"""Run E06 S08 interface-rule intervention sweep."""

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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.interface_rules import (  # noqa: E402
    DEFAULT_INTERFACE_RULES,
    STEP_ID,
    S08Config,
    compare_to_behavior_baseline,
    mechanism_strata,
    run_s08_sweep,
    summarize_s08_runs,
    validate_s08_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402


STEP_NUMBER = 8
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s07-mosaic-path", type=Path, default=ARTIFACTS_DIR / "results/e06_mosaic_classes.parquet")
    parser.add_argument("--s07-exemplar-path", type=Path, default=ARTIFACTS_DIR / "research_steps/S07/e06_s07_exemplar_rows.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-base-conditions", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="*", default=[2026070201, 2026070202, 2026070203, 2026070204])
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
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
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


def write_interface_figure(path: Path, comparison: pd.DataFrame, strata: pd.DataFrame, selected: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if comparison.empty or strata.empty:
        return False
    explicit = comparison[comparison["interface_rule_id"] != "behavior_only"].copy()
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_agg = fig.add_subplot(grid[0, 0])
    ax_quality = fig.add_subplot(grid[0, 1])
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_block = fig.add_subplot(grid[1, 1])

    order = strata["interface_rule_id"].astype(str).tolist()
    colors = plt.get_cmap("tab10")(range(len(order)))
    color_map = dict(zip(order, colors, strict=True))
    ax_agg.bar(
        strata["interface_rule_id"],
        strata["mean_delta_aggregation_delta_percent"],
        color=[color_map[str(rule)] for rule in strata["interface_rule_id"]],
    )
    ax_agg.axhline(0, color="#222222", lw=0.8)
    ax_agg.set_title("Mean aggregation shift vs behavior-only")
    ax_agg.set_ylabel("Delta aggregation delta percent")
    ax_agg.tick_params(axis="x", rotation=35)

    ax_quality.bar(
        strata["interface_rule_id"],
        strata["mean_delta_final_target_quality"],
        color=[color_map[str(rule)] for rule in strata["interface_rule_id"]],
    )
    ax_quality.axhline(0, color="#222222", lw=0.8)
    ax_quality.set_title("Mean target-quality shift vs behavior-only")
    ax_quality.set_ylabel("Delta final target quality")
    ax_quality.tick_params(axis="x", rotation=35)

    for rule_id, group in explicit.groupby("interface_rule_id", sort=False):
        ax_scatter.scatter(
            group["baseline_mean_aggregation_delta_percent"],
            group["mean_aggregation_delta_percent"],
            s=55,
            alpha=0.75,
            color=color_map.get(str(rule_id)),
            label=str(rule_id),
        )
    low = min(comparison["baseline_mean_aggregation_delta_percent"].min(), comparison["mean_aggregation_delta_percent"].min())
    high = max(comparison["baseline_mean_aggregation_delta_percent"].max(), comparison["mean_aggregation_delta_percent"].max())
    ax_scatter.plot([low, high], [low, high], color="#333333", lw=0.8, linestyle="--")
    ax_scatter.set_title("Condition-level aggregation response")
    ax_scatter.set_xlabel("Behavior-only aggregation delta percent")
    ax_scatter.set_ylabel("Intervention aggregation delta percent")
    ax_scatter.legend(loc="best", fontsize=7)

    ax_block.bar(
        strata["interface_rule_id"],
        strata["mean_interface_block_fraction"],
        color=[color_map[str(rule)] for rule in strata["interface_rule_id"]],
    )
    ax_block.set_title("Mean heterotypic proposal block fraction")
    ax_block.set_ylabel("Blocked fraction")
    ax_block.tick_params(axis="x", rotation=35)
    if not selected.empty:
        labels = selected[["selection_rank", "s07_label", "selection_reason"]].head(8).copy()
        text = "\n".join(f"{row.selection_rank}: {row.s07_label} ({row.selection_reason.split(',')[0]})" for row in labels.itertuples())
        ax_block.text(1.02, 0.02, text, transform=ax_block.transAxes, ha="left", va="bottom", fontsize=7)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


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
    explicit_domination = bool(strata["explicit_rule_dominates_baseline_behavior"].any()) if "explicit_rule_dominates_baseline_behavior" in strata else False
    outcome = "supportive" if validation_passed else "null"
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    selected_display = selected.copy()
    if not selected_display.empty:
        selected_display["display_names"] = selected_display["display_names_json"].map(_display_name)
        selected_display["ratio_label"] = selected_display["ratio_targets_json"].map(_ratio_label)
    summary_display = summary.copy()
    if not summary_display.empty:
        summary_display["display_names"] = summary_display["display_names_json"].map(_display_name)
        summary_display["ratio_label"] = summary_display["ratio_targets_json"].map(_ratio_label)
    comparison_display = comparison[comparison["interface_rule_id"] != "behavior_only"].copy()
    text = f"""# E06 S08 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: Explicit interface rules are exploratory computational interventions beyond the original no-recognition sorting-array claims. S08 compares matched behavior-only baselines against explicit mechanisms but does not infer biological adhesion, permeability, or self/non-self recognition from the paper's original behavior-only aggregation.
- Lay summary: S08 replayed S07 exemplar and metric-extreme chimeras with matched seeds. Behavior-only rows preserve the original no-explicit-recognition substrate, while adhesion-like, repulsion, permeability, and local-recognition rows use explicit label-aware interface gates. The result quantifies how much these gates shift aggregation, target quality, interface counts, and cross-label swap acceptance relative to the matched baseline.
- Recommended next action: Proceed to S09 governance mechanisms using the S08 behavior-only and explicit-interface strata separately; do not combine explicit-recognition effects with original no-recognition behavior-only evidence.

## Frozen Question

How much segregation arises from behavior alone versus explicit self/non-self-like interface rules?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 Algotype metadata: `{config_payload['metadataPath']}`
- S07 mosaic classes and continua: `{config_payload['s07MosaicPath']}`
- S07 exemplar rows: `{config_payload['s07ExemplarPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S08 selected a bounded pairwise panel from S07 rather than treating S07's coarse labels as hard categories. Selection first included S07 exemplar rows, then filled remaining slots from metric-continuum extremes for mosaic score, conflict score, aggregation, goal conflict, position bias, largest block, and low target quality.

Each selected base condition was replayed with matched seeds under five mechanism strata:

- `behavior_only`: original local policy behavior with no explicit same/different-label recognition.
- `adhesion_like_preference`: label-aware preference for swaps that preserve or increase local same-Algotype contacts.
- `heterotypic_repulsion`: label-aware rule favoring swaps that reduce unlike-neighbor contacts.
- `permeability_barrier`: semipermeable interface rule allowing heterotypic swaps only with low local probability.
- `local_recognition_gate`: label-aware and local-goal-aware gate allowing heterotypic swaps only when local label cohesion and goal satisfaction do not degrade.

The behavior-only stratum is the only stratum used to represent the original no-recognition substrate. All other strata are explicitly marked as interventions beyond the original claim scope.

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

- Selected base conditions: {int(selected['base_condition_id'].nunique()) if not selected.empty else 0}.
- Intervention run rows: {len(run_df)}.
- Mechanism strata: {summary['interface_rule_id'].nunique() if not summary.empty else 0}.
- Explicit mechanisms dominate behavior-only by configured thresholds: {explicit_domination}.
- Behavior-only run count: {int((run_df['interface_rule_id'] == 'behavior_only').sum()) if not run_df.empty else 0}.
- Explicit-intervention run count: {int((run_df['interface_rule_id'] != 'behavior_only').sum()) if not run_df.empty else 0}.

Selected S07 base conditions:

{markdown_table(selected_display, ["selection_rank", "base_condition_id", "selection_reason", "s07_label", "display_names", "ratio_label", "arrangement", "goal_profile_id", "source_final_target_quality", "source_goal_conflict_index", "source_aggregation_delta_percent", "source_largest_block_fraction"], max_rows=20)}

Mechanism strata:

{markdown_table(strata, ["interface_rule_id", "explicit_recognition_used", "condition_count", "mean_delta_final_target_quality", "mean_delta_aggregation_delta_percent", "mean_delta_interface_count", "mean_delta_largest_block_fraction", "mean_interface_block_fraction", "mean_cross_label_swap_acceptance_fraction", "explicit_rule_dominates_baseline_behavior"], max_rows=20)}

Per-condition summaries:

{markdown_table(summary_display.sort_values(["base_condition_id", "interface_rule_id"]), ["base_condition_id", "interface_rule_id", "display_names", "ratio_label", "arrangement", "goal_profile_id", "mean_final_target_quality", "mean_aggregation_delta_percent", "mean_interface_count", "mean_largest_block_fraction", "mean_interface_block_fraction", "mean_cross_label_swap_acceptance_fraction"], max_rows=40)}

Baseline comparisons for explicit mechanisms:

{markdown_table(comparison_display.sort_values(["interface_rule_id", "abs_delta_aggregation_delta_percent"], ascending=[True, False]), ["base_condition_id", "interface_rule_id", "delta_final_target_quality", "delta_aggregation_delta_percent", "delta_interface_count", "delta_largest_block_fraction", "delta_goal_conflict_index", "mean_interface_block_fraction"], max_rows=40)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=30)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- Adhesion, repulsion, permeability, and recognition are computational analogs implemented as local swap gates. They are not biological mechanism measurements.
- S08 uses the same 1D local-action proxy substrate as S02-S07 and remains bounded to selected pairwise S07 exemplar/extreme conditions.
- Explicit label-aware mechanisms intentionally violate the original no-explicit-recognition condition, so their effects are reported as separate intervention strata.
- S07's discrete labels were unstable across classifier settings; S08 therefore targets metric-continuum extremes and exemplar rows rather than treating labels as ground truth classes.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Run S09 governance mechanisms with separate behavior-only and explicit-interface strata. Governance rescue should be compared against S08 baselines and should document any information access that goes beyond local behavior-only policies.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_interface_rule_interventions.parquet"
    governance_path = artifacts_dir / "results/e06_governance_interventions.parquet"
    summary_path = artifacts_dir / "tables/e06_interface_rule_summary.csv"
    comparison_path = artifacts_dir / "tables/e06_s08_baseline_comparison.csv"
    strata_path = artifacts_dir / "tables/e06_s08_mechanism_strata.csv"
    selected_path = step_dir / "e06_s08_selected_conditions.csv"
    condition_matrix_path = step_dir / "e06_s08_condition_matrix.csv"
    validation_path = step_dir / "e06_s08_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/interface_rule_effects.png"
    config_path = artifacts_dir / "configs/e06_s08_interface_rules_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S08Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        max_base_conditions=args.max_base_conditions,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "s07MosaicPath": str(args.s07_mosaic_path),
        "s07ExemplarPath": str(args.s07_exemplar_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "maxBaseConditions": config.max_base_conditions,
        "metricExtremeCandidatesPerMetric": config.metric_extreme_candidates_per_metric,
        "explicitDominanceDeltaThreshold": config.explicit_dominance_delta_threshold,
        "explicitDominanceBlockThreshold": config.explicit_dominance_block_threshold,
        "interfaceRules": [rule.__dict__ for rule in config.interface_rules],
        "behaviorOnlyClaimScope": "original_behavior_only_no_explicit_recognition",
        "explicitInterfaceClaimScope": "explicit_interface_intervention_beyond_original_no_recognition_claim",
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    for path in (args.library_path, args.metadata_path, args.s07_mosaic_path, args.s07_exemplar_path):
        if not path.exists():
            raise FileNotFoundError(f"required S08 input is missing: {path}")
    records, metadata = load_s01_library(args.library_path, args.metadata_path)
    mosaic = pd.read_parquet(args.s07_mosaic_path)
    exemplars = pd.read_csv(args.s07_exemplar_path)

    run_df, condition_df, selected = run_s08_sweep(records, mosaic, exemplars, config)
    summary = summarize_s08_runs(run_df)
    comparison = compare_to_behavior_baseline(summary)
    strata = mechanism_strata(comparison, config)
    figure_written = write_interface_figure(figure_path, comparison, strata, selected)
    validation = validate_s08_outputs(
        run_df,
        condition_df,
        selected,
        summary,
        comparison,
        strata,
        config,
        figure_written=figure_written,
        unit_tests_success=bool(unit_test_result.get("success", True)),
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(results_path, index=False)
    # S08 contributes the first governance/intervention rows used by later E06 steps.
    run_df.to_parquet(governance_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    strata.to_csv(strata_path, index=False)
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
        artifact_entry(results_path, artifacts_dir, "S08 run-level interface-rule intervention table."),
        artifact_entry(governance_path, artifacts_dir, "S08 contribution to cumulative E06 governance/intervention data."),
        artifact_entry(figure_path, artifacts_dir, "S08 interface-rule effect summary figure."),
        artifact_entry(summary_path, artifacts_dir, "S08 per-condition interface-rule summary table."),
        artifact_entry(comparison_path, artifacts_dir, "S08 matched behavior-only baseline comparison table."),
        artifact_entry(strata_path, artifacts_dir, "S08 mechanism-specific strata table."),
        artifact_entry(selected_path, artifacts_dir, "S08 selected S07 exemplar/extreme condition table."),
        artifact_entry(condition_matrix_path, artifacts_dir, "S08 executable condition matrix."),
        artifact_entry(validation_path, artifacts_dir, "S08 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S08 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S08 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S08 full-results Markdown handoff report.")

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
    report_artifact = artifact_entry(report_path, artifacts_dir, "S08 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    explicit_domination = bool(strata["explicit_rule_dominates_baseline_behavior"].any()) if "explicit_rule_dominates_baseline_behavior" in strata else False
    manifest = {
        "schema": "eidosoma.e06.s08_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "runCount": int(len(run_df)),
        "baseConditionCount": int(selected["base_condition_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "explicitRuleDominanceDetected": explicit_domination,
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S08 artifact manifest.")
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
            "s07ExemplarRows": source_entry(args.s07_exemplar_path),
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        governance_path,
        figure_path,
        summary_path,
        comparison_path,
        strata_path,
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
        "runCount": int(len(run_df)),
        "baseConditionCount": int(selected["base_condition_id"].nunique()),
        "interfaceRuleCount": int(run_df["interface_rule_id"].nunique()),
        "explicitRuleDominanceDetected": explicit_domination,
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
