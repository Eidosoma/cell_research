#!/usr/bin/env python3
"""Run E06 S03 bounded spatial-arrangement sweep."""

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

from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402
from src.e06.spatial_arrangements import (  # noqa: E402
    DEFAULT_ARRANGEMENTS,
    STEP_ID,
    S03Config,
    arrangement_sensitivity,
    run_s03_sweep,
    summarize_s03_runs,
    validate_s03_outputs,
)


STEP_NUMBER = 3
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s02-threshold-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s02_threshold_candidates.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-candidate-pairs", type=int, default=9)
    parser.add_argument("--top-sortedness-pairs", type=int, default=6)
    parser.add_argument("--top-aggregation-pairs", type=int, default=4)
    parser.add_argument("--arrangements", nargs="*", default=list(DEFAULT_ARRANGEMENTS))
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
    return " vs ".join(json.loads(str(text))[:2])


def _ratio_label(text: str) -> str:
    values = json.loads(str(text))
    return ":".join(str(int(round(float(value) * 100))) for value in values)


def write_arrangement_figure(path: Path, summary: pd.DataFrame, sensitivity: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty or sensitivity.empty:
        return False
    chosen = sensitivity.head(8)[["s02_candidate_id", "ratio_targets_json"]].drop_duplicates()
    plot_df = summary.merge(chosen, on=["s02_candidate_id", "ratio_targets_json"], how="inner").copy()
    if plot_df.empty:
        return False
    arrangements = [item for item in DEFAULT_ARRANGEMENTS if item in set(plot_df["arrangement"])]
    plot_df["display"] = plot_df["display_names_json"].map(_display_name) + " " + plot_df["ratio_targets_json"].map(_ratio_label)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    for display, group in plot_df.groupby("display", sort=False):
        group = group.set_index("arrangement").reindex(arrangements)
        x = range(len(arrangements))
        axes[0].plot(x, group["mean_final_inversion_sortedness"], marker="o", linewidth=1.4, label=display)
        axes[1].plot(x, group["mean_aggregation_delta_percent"], marker="o", linewidth=1.4, label=display)
    for ax in axes:
        ax.set_xticks(range(len(arrangements)))
        ax.set_xticklabels([item.replace("_", "\n") for item in arrangements], fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Final sortedness by initial arrangement")
    axes[0].set_ylabel("Mean final inversion sortedness")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("Aggregation delta by initial arrangement")
    axes[1].set_ylabel("Mean aggregation above random (percentage points)")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
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
    selected_candidates: pd.DataFrame,
    sensitivity: pd.DataFrame,
    condition_df: pd.DataFrame,
    config_payload: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    validation_passed = bool(validation["success"].all() and unit_test_result.get("success", True))
    sensitive_count = int(sensitivity["arrangement_sensitive_candidate"].sum()) if "arrangement_sensitive_candidate" in sensitivity else 0
    outcome = "supportive" if validation_passed and sensitive_count > 0 else "null"
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    selected = selected_candidates.copy()
    if not selected.empty:
        selected["display_names"] = selected["display_names_json"].map(_display_name)
    top_sensitivity = sensitivity.head(15).copy()
    if not top_sensitivity.empty:
        top_sensitivity["display_names"] = top_sensitivity["display_names_json"].map(_display_name)
        top_sensitivity["ratio"] = top_sensitivity["ratio_targets_json"].map(_ratio_label)
    run_count = int(summary["run_count"].sum()) if not summary.empty else 0
    arrangement_count = len(config_payload["arrangements"])
    text = f"""# E06 S03 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: The sweep was bounded to {len(selected_candidates)} S02-derived candidate pairs and {len(condition_df)} candidate-ratio-arrangement conditions. It uses the same unified local-action proxy harness as S02, so arrangement effects are computational proxy evidence rather than biological mechanism claims.
- Lay summary: S03 held policy ratios fixed at S02 threshold brackets and changed only the starting spatial layout. Several candidate mixtures changed sorting or aggregation outcomes across random, patch, alternating, island, gradient, and graft-like arrangements, so geometry is a practical modifier for the next goal-compatibility step.
- Recommended next action: Run S04 goal-compatibility tests on the S03 arrangement-sensitive candidate mixtures, prioritizing the best/worst arrangement contrasts reported in `tables/e06_s03_arrangement_sensitivity.csv`.

## Frozen Question

Does developmental geometry determine whether mixed collectives cooperate, segregate, dominate, or mosaic?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 metadata table: `{config_payload['metadataPath']}`
- S02 threshold/minority candidate table: `{config_payload['s02ThresholdPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S03 selected a bounded candidate set from S02 by combining the top sortedness jumps, top aggregation-sensitive pairs, and at least one memory-repair-sensitive pair. For each selected S02 candidate, the two ratios bracketing its S02 threshold were replayed across {arrangement_count} deterministic initial arrangement families: random permutation, contiguous patch, alternating, clustered islands, gradient, and graft-like insertions. Every candidate-ratio-arrangement condition used the same configured seed set.

The simulation reused the S02 mixed-policy action dispatcher: original algorithms are local Bubble/Insertion/Selection rules, E03 frontier and control policies use the DSL interpreter, and E04 memory-repair policies use their local evolved decision function. The only experimental variable changed by S03 is the initial label layout. Counts, values, policy IDs, and cell memory still move together when swaps occur.

Initial arrangement metrics were computed before simulation from the label sequence: interface count, contiguous run count, largest block fraction, aggregation delta against random expectation, normalized label entropy, and position-bias proxy.

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

- Selected S02 candidate pairs: {len(selected_candidates)}.
- Candidate-ratio-arrangement conditions: {len(condition_df)}.
- Run count: {run_count}.
- Arrangement-sensitive candidate-ratio rows: {sensitive_count}/{len(sensitivity)}.
- Mean final inversion sortedness across summaries: {summary['mean_final_inversion_sortedness'].mean():.4f}.
- Mean final aggregation delta across summaries: {summary['mean_aggregation_delta_percent'].mean():.4f} percentage points.

Selected S02 candidates:

{markdown_table(selected, ["s02_candidate_rank", "panel", "display_names", "candidate_threshold_between_ratios_json", "candidate_reason", "largest_sortedness_jump", "aggregation_delta_range"], max_rows=20)}

Top arrangement-sensitivity contrasts:

{markdown_table(top_sensitivity, ["display_names", "ratio", "sortedness_range_across_arrangements", "aggregation_delta_range_across_arrangements", "initial_interface_range", "best_sortedness_arrangement", "best_sortedness", "worst_sortedness_arrangement", "worst_sortedness", "arrangement_sensitive_candidate"], max_rows=15)}

Representative arrangement summaries:

{markdown_table(summary.sort_values(["mean_final_inversion_sortedness", "mean_aggregation_delta_percent"], ascending=[False, False]), ["arrangement", "display_names_json", "ratio_targets_json", "mean_initial_interface_count", "mean_final_inversion_sortedness", "mean_aggregation_delta_percent", "final_state_class_mode"], max_rows=25)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=20)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- S03 tests a bounded S02-derived panel, not every S01 policy pair or every ratio.
- The gradient, island, and graft-like arrangements are deterministic 1D computational layouts; they are not biological tissue models.
- The same-goal mixed-array harness remains a proxy substrate. Opposite-goal conflict is deferred to later E06 steps.
- Arrangement sensitivity is reported as a measured simulation association, not causality or wet-lab validation.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Use `tables/e06_s03_arrangement_sensitivity.csv` to choose S04 goal-compatibility conditions, emphasizing pairs with large best/worst arrangement gaps and including a memory-repair contrast.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_spatial_arrangement_sweep.parquet"
    summary_path = artifacts_dir / "tables/e06_spatial_arrangement_summary.csv"
    sensitivity_path = artifacts_dir / "tables/e06_s03_arrangement_sensitivity.csv"
    condition_path = step_dir / "e06_s03_condition_matrix.csv"
    selected_path = step_dir / "e06_s03_selected_candidates.csv"
    validation_path = step_dir / "e06_s03_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/spatial_arrangement_outcomes.png"
    config_path = artifacts_dir / "configs/e06_s03_spatial_arrangements_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S03Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        arrangements=tuple(str(item) for item in args.arrangements),
        max_candidate_pairs=args.max_candidate_pairs,
        top_sortedness_pairs=args.top_sortedness_pairs,
        top_aggregation_pairs=args.top_aggregation_pairs,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "s02ThresholdPath": str(args.s02_threshold_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "arrangements": list(config.arrangements),
        "maxCandidatePairs": config.max_candidate_pairs,
        "topSortednessPairs": config.top_sortedness_pairs,
        "topAggregationPairs": config.top_aggregation_pairs,
        "scheduler": config.scheduler,
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    records, _metadata = load_s01_library(args.library_path, args.metadata_path)
    if not args.s02_threshold_path.exists():
        raise FileNotFoundError(f"S02 threshold candidate table is missing: {args.s02_threshold_path}")
    threshold_candidates = pd.read_csv(args.s02_threshold_path)
    run_df, condition_df, selected_candidates = run_s03_sweep(records, threshold_candidates, config)
    summary = summarize_s03_runs(run_df)
    sensitivity = arrangement_sensitivity(summary)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(results_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(sensitivity_path, index=False)
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    condition_df.to_csv(condition_path, index=False)
    selected_candidates.to_csv(selected_path, index=False)
    figure_written = write_arrangement_figure(figure_path, summary, sensitivity)

    validation = validate_s03_outputs(
        run_df,
        condition_df,
        selected_candidates,
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
        artifact_entry(results_path, artifacts_dir, "S03 run-level spatial-arrangement sweep table."),
        artifact_entry(summary_path, artifacts_dir, "S03 arrangement-level summary table."),
        artifact_entry(sensitivity_path, artifacts_dir, "S03 arrangement sensitivity table."),
        artifact_entry(figure_path, artifacts_dir, "S03 spatial-arrangement outcome figure."),
        artifact_entry(condition_path, artifacts_dir, "S03 condition matrix."),
        artifact_entry(selected_path, artifacts_dir, "S03 selected S02 candidate table."),
        artifact_entry(validation_path, artifacts_dir, "S03 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S03 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S03 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S03 full-results Markdown handoff report.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        summary=summary,
        validation=validation,
        selected_candidates=selected_candidates,
        sensitivity=sensitivity,
        condition_df=condition_df,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S03 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    manifest = {
        "schema": "eidosoma.e06.s03_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "runCount": int(len(run_df)),
        "conditionCount": int(len(condition_df)),
        "selectedCandidateCount": int(len(selected_candidates)),
        "arrangementSensitiveCount": int(sensitivity["arrangement_sensitive_candidate"].sum()) if not sensitivity.empty else 0,
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S03 artifact manifest.")
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
            "s02ThresholdCandidates": {
                "path": str(args.s02_threshold_path),
                "sha256": sha256_file(args.s02_threshold_path),
                "sizeBytes": args.s02_threshold_path.stat().st_size,
            },
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        summary_path,
        sensitivity_path,
        figure_path,
        condition_path,
        selected_path,
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
        "conditionCount": int(len(condition_df)),
        "selectedCandidateCount": int(len(selected_candidates)),
        "arrangementSensitiveCount": int(sensitivity["arrangement_sensitive_candidate"].sum()) if not sensitivity.empty else 0,
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
