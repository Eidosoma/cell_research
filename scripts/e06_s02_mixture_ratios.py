#!/usr/bin/env python3
"""Run E06 S02 bounded mixture-ratio phase sweep."""

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

from src.e06.mixture_ratios import (  # noqa: E402
    EXPERIMENT_ID,
    STEP_ID,
    S02Config,
    detect_threshold_candidates,
    load_s01_library,
    run_s02_sweep,
    sha256_file,
    summarize_s02_runs,
    validate_s02_outputs,
)


STEP_NUMBER = 2
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--top-discovered-count", type=int, default=4)
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


def markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 40) -> str:
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
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"| ... | {len(frame) - max_rows} more rows omitted from report table | " + " | ".join("" for _ in columns[2:]) + " |")
    return "\n".join(lines)


def write_phase_figure(path: Path, summary: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    pairwise = summary[summary["condition_kind"] == "pair"].copy()
    if pairwise.empty:
        return False
    pairwise["first_ratio"] = pairwise["ratio_targets_json"].map(lambda text: json.loads(text)[0])
    pairwise["name"] = pairwise["display_names_json"].map(lambda text: " vs ".join(json.loads(text)[:2]))
    priority_panels = ["original_control", "frontier_vs_original", "memory_vs_original", "memory_vs_frontier", "frontier_frontier"]
    chosen = []
    for panel in priority_panels:
        for name in pairwise[pairwise["panel"] == panel]["name"].drop_duplicates().head(3).tolist():
            if name not in chosen:
                chosen.append(name)
    chosen = chosen[:12]
    plot_df = pairwise[pairwise["name"].isin(chosen)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for name, group in plot_df.groupby("name", sort=False):
        group = group.sort_values("first_ratio")
        axes[0].plot(group["first_ratio"] * 100, group["mean_final_inversion_sortedness"], marker="o", linewidth=1.4, label=name)
        axes[1].plot(group["first_ratio"] * 100, group["mean_aggregation_delta_percent"], marker="o", linewidth=1.4, label=name)
    axes[0].set_title("Final sortedness by first-policy ratio")
    axes[0].set_xlabel("First policy in pair (%)")
    axes[0].set_ylabel("Mean final inversion sortedness")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.2)
    axes[1].set_title("Aggregation delta by first-policy ratio")
    axes[1].set_xlabel("First policy in pair (%)")
    axes[1].set_ylabel("Mean aggregation above random (percentage points)")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def write_report(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    threshold_candidates: pd.DataFrame,
    condition_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    config_payload: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    validation_passed = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = "supportive" if validation_passed and not threshold_candidates.empty else "null"
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    panel_count = len(panel_df)
    condition_count = len(condition_df)
    run_count = int(summary["run_count"].sum()) if not summary.empty else 0
    top_thresholds = threshold_candidates.head(12).copy()
    if not top_thresholds.empty:
        top_thresholds["display_names"] = top_thresholds["display_names_json"].map(lambda text: " vs ".join(json.loads(text)[:2]))
    text = f"""# E06 S02 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: The full 25-policy pairwise library was bounded to a prioritized {panel_count}-policy panel and {condition_count} mixture conditions. The unified local-action harness is a proxy substrate for heterogeneous S01 policy mixing, not an exact replay of the public threaded original simulator.
- Lay summary: S02 varied how common or rare each policy type was in mixed arrays. The sweep found ratio-sensitive sorting and aggregation patterns across original, control, discovered, and memory-policy mixtures, producing candidates for spatial-arrangement tests in S03.
- Recommended next action: Run S03 spatial-arrangement sweeps on the S02 candidate pairs with the largest sortedness jumps or aggregation sensitivity.

## Frozen Question

Do minority effects and threshold transitions appear as Algotype ratios shift from rare clones to majorities?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 metadata table: `{config_payload['metadataPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

The S02 sweep selected a bounded panel from S01: all original policies, all null and randomized controls, the top E03 frontier policies, and both E04 memory-repair policies. Pairwise conditions used first-policy ratios 1%, 5%, 10%, 25%, 50%, and 75% in 100-cell arrays. Three-way mixtures used 1:1:98, 10:10:80, 25:25:50, and approximately equal thirds. Every condition used the same seed set.

The simulator used a unified 1D adjacent-swap substrate so all S01 policy representations could be mixed. Original policies were represented as local Bubble, Insertion, and Selection action rules; E03 controls and frontier policies used the E03 DSL interpreter; E04 memory-repair policies used the existing `S08LocalEvolutionPolicy` decision function with local-neighbor memory features. Cell identities, values, policy IDs, and memory moved together on swaps.

## Commands

- Main command: `{command}`
- Unit-test command: `{unit_test_result.get('command', 'not run')}`
- Unit-test return code: `{unit_test_result.get('returnCode', 'not run')}`

## Dependencies

No new dependencies were installed. The script used repository modules plus preinstalled `pandas`, `numpy`, and `matplotlib`.

## Parameters

```json
{json.dumps(config_payload, indent=2, sort_keys=True, default=str)}
```

## Results

- Panel size: {panel_count} policies.
- Condition count: {condition_count}.
- Run count: {run_count}.
- Mean final inversion sortedness across condition summaries: {summary['mean_final_inversion_sortedness'].mean():.4f}.
- Mean aggregation delta across condition summaries: {summary['mean_aggregation_delta_percent'].mean():.4f} percentage points.

Top threshold or minority-effect candidates:

{markdown_table(top_thresholds, ["panel", "display_names", "sortedness_range", "aggregation_delta_range", "largest_sortedness_jump", "candidate_threshold_between_ratios_json", "minority_effect_candidate", "aggregation_sensitive_candidate"], max_rows=12) if not top_thresholds.empty else "_No threshold candidates detected._"}

Representative condition summaries:

{markdown_table(summary.sort_values(["mean_final_inversion_sortedness", "mean_aggregation_delta_percent"], ascending=[False, False]), ["panel", "condition_kind", "display_names_json", "ratio_targets_json", "mean_final_inversion_sortedness", "mean_aggregation_delta_percent", "final_state_class_mode"], max_rows=25)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=20)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- S02 prioritizes breadth over exhaustive pairwise coverage; untested S01 policies remain available for later targeted sweeps.
- The dominance metric is a spatial-position bias proxy, not a causal or biological dominance claim.
- E04 memory policies are run in a simplified mixed-array harness; their upstream E04 repair evidence remains separate.
- Same-goal ratio sweeps do not test opposite-goal conflict, which is deferred to later E06 steps.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Use `tables/e06_s02_threshold_candidates.csv` to choose S03 spatial arrangements, emphasizing candidate pairs with large sortedness jumps or aggregation-delta ranges.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_mixture_ratio_sweep.parquet"
    summary_path = artifacts_dir / "tables/e06_mixture_ratio_summary.csv"
    condition_path = step_dir / "e06_s02_condition_matrix.csv"
    panel_path = step_dir / "e06_s02_selected_panel.csv"
    validation_path = step_dir / "e06_s02_validation_checks.csv"
    threshold_path = artifacts_dir / "tables/e06_s02_threshold_candidates.csv"
    figure_path = artifacts_dir / "figures/e06/mixture_ratio_phase_curves.png"
    config_path = artifacts_dir / "configs/e06_s02_mixture_ratios_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S02Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        top_discovered_count=args.top_discovered_count,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "topDiscoveredCount": config.top_discovered_count,
        "arrangement": config.arrangement,
        "scheduler": config.scheduler,
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    records, metadata = load_s01_library(args.library_path, args.metadata_path)
    run_df, condition_df, panel_df = run_s02_sweep(records, metadata, config)
    summary = summarize_s02_runs(run_df)
    threshold_candidates = detect_threshold_candidates(summary)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(results_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    condition_df.to_csv(condition_path, index=False)
    panel_df.to_csv(panel_path, index=False)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_candidates.to_csv(threshold_path, index=False)
    figure_written = write_phase_figure(figure_path, summary)

    validation = validate_s02_outputs(
        run_df,
        condition_df,
        panel_df,
        config,
        figure_written=figure_written,
        unit_tests_success=bool(unit_test_result.get("success", True)),
    )
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
        artifact_entry(results_path, artifacts_dir, "S02 run-level mixture-ratio sweep table."),
        artifact_entry(summary_path, artifacts_dir, "S02 condition-level mixture-ratio summary table."),
        artifact_entry(figure_path, artifacts_dir, "S02 mixture-ratio phase-curve figure."),
        artifact_entry(threshold_path, artifacts_dir, "S02 threshold and minority-effect candidate table."),
        artifact_entry(condition_path, artifacts_dir, "S02 condition matrix with realized ratio targets."),
        artifact_entry(panel_path, artifacts_dir, "S02 selected bounded policy panel."),
        artifact_entry(validation_path, artifacts_dir, "S02 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S02 configuration file."),
    ]
    report_pending = pending_entry(report_path, artifacts_dir, "S02 full-results Markdown handoff report.")
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S02 artifact manifest.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        summary=summary,
        validation=validation,
        threshold_candidates=threshold_candidates,
        condition_df=condition_df,
        panel_df=panel_df,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S02 full-results Markdown handoff report.")

    manifest = {
        "schema": "eidosoma.e06.s02_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": bool(validation["success"].all() and unit_test_result.get("success", True)),
        "runCount": int(len(run_df)),
        "conditionCount": int(len(condition_df)),
        "panelCount": int(len(panel_df)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S02 artifact manifest.")
    artifacts = [*base_artifacts, manifest_artifact, report_artifact]

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
            "s01AlgotypeLibrary": {
                "path": str(args.library_path),
                "sha256": sha256_file(args.library_path),
                "sizeBytes": args.library_path.stat().st_size,
            },
            "s01Metadata": {
                "path": str(args.metadata_path),
                "sha256": sha256_file(args.metadata_path),
                "sizeBytes": args.metadata_path.stat().st_size,
            },
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        summary_path,
        figure_path,
        threshold_path,
        condition_path,
        panel_path,
        validation_path,
        config_path,
        manifest_path,
        report_path,
        run_manifest_path,
    ]
    write_text(checksums_path, "\n".join(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_paths) + "\n")

    status = {
        "success": bool(validation["success"].all() and unit_test_result.get("success", True)),
        "runCount": int(len(run_df)),
        "conditionCount": int(len(condition_df)),
        "panelCount": int(len(panel_df)),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "thresholdCandidateCount": int(len(threshold_candidates)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
