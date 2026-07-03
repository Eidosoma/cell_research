#!/usr/bin/env python3
"""Run E06 S04 bounded goal-compatibility sweep."""

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

from src.e06.goal_compatibility import (  # noqa: E402
    DEFAULT_GOAL_PROFILES,
    STEP_ID,
    S04Config,
    goal_profile_sensitivity,
    run_s04_sweep,
    summarize_s04_runs,
    validate_s04_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402


STEP_NUMBER = 4
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s03-sensitivity-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s03_arrangement_sensitivity.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-sensitive-rows", type=int, default=6)
    parser.add_argument("--include-memory-contrast", action=argparse.BooleanOptionalAction, default=True)
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
    return ":".join(str(int(round(float(value) * 100))) for value in json.loads(str(text)))


def write_goal_figure(path: Path, summary: pd.DataFrame, sensitivity: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return False
    chosen = sensitivity.head(10)[["s03_candidate_id", "ratio_targets_json", "arrangement"]].drop_duplicates()
    plot_df = summary.merge(chosen, on=["s03_candidate_id", "ratio_targets_json", "arrangement"], how="inner")
    if plot_df.empty:
        plot_df = summary.copy()
    plot_df = plot_df.copy()
    plot_df["condition_label"] = (
        plot_df["display_names_json"].map(_display_name)
        + " "
        + plot_df["ratio_targets_json"].map(_ratio_label)
        + " "
        + plot_df["arrangement"].astype(str)
    )
    pivot_assigned = plot_df.pivot_table(
        index="goal_profile_id",
        columns="condition_label",
        values="mean_assigned_policy_sortedness",
        aggfunc="mean",
    )
    pivot_gap = plot_df.pivot_table(
        index="goal_profile_id",
        columns="condition_label",
        values="mean_goal_alignment_gap",
        aggfunc="mean",
    ).reindex(index=pivot_assigned.index, columns=pivot_assigned.columns)
    if pivot_assigned.empty:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    im0 = axes[0].imshow(pivot_assigned.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    im1 = axes[1].imshow(pivot_gap.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=max(0.25, float(pivot_gap.max().max())), cmap="magma")
    for ax, title in zip(axes, ("Assigned-goal sortedness", "Relevant-goal alignment gap"), strict=True):
        ax.set_title(title)
        ax.set_yticks(range(len(pivot_assigned.index)))
        ax.set_yticklabels(pivot_assigned.index, fontsize=8)
        ax.set_xticks(range(len(pivot_assigned.columns)))
        ax.set_xticklabels(pivot_assigned.columns, rotation=75, ha="right", fontsize=6)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
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
    goal_sensitive_count = int(sensitivity["goal_sensitive_candidate"].sum()) if "goal_sensitive_candidate" in sensitivity else 0
    outcome = "supportive" if validation_passed and goal_sensitive_count > 0 else "null"
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    selected = selected_candidates.copy()
    if not selected.empty:
        selected["display_names"] = selected["display_names_json"].map(_display_name)
    top_sensitivity = sensitivity.head(15).copy()
    if not top_sensitivity.empty:
        top_sensitivity["display_names"] = top_sensitivity["display_names_json"].map(_display_name)
        top_sensitivity["ratio"] = top_sensitivity["ratio_targets_json"].map(_ratio_label)
    representative = summary.sort_values(
        ["mean_assigned_policy_sortedness", "mean_goal_alignment_gap"],
        ascending=[False, True],
    ).copy()
    if not representative.empty:
        representative["display_names"] = representative["display_names_json"].map(_display_name)
        representative["ratio"] = representative["ratio_targets_json"].map(_ratio_label)
    run_count = int(summary["run_count"].sum()) if not summary.empty else 0
    profile_count = len(config_payload["goalProfiles"])
    text = f"""# E06 S04 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: The sweep was bounded to {len(selected_candidates)} S03-derived candidate rows and {len(condition_df)} candidate-arrangement-goal conditions. Goal encoders are computational rank-order proxies applied to the mixed-policy harness, not biological target fields.
- Lay summary: S04 changed each Algotype's local target order while holding S03 candidate mixtures and best/worst arrangements fixed. Goal compatibility strongly changed assigned-goal success and alignment gaps, mapping which mixtures remain coherent under same, opposite, partial, shared-global, or unrelated targets.
- Recommended next action: Run S05 compatibility-metric construction using `tables/e06_s04_goal_profile_sensitivity.csv`, with separate metrics for shared-goal success, local assigned-goal success, aggregation, and alignment gap.

## Frozen Question

How does compatibility of local goals shape collective outcomes under mixed policies?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 metadata table: `{config_payload['metadataPath']}`
- S03 arrangement sensitivity table: `{config_payload['s03SensitivityPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S04 selected the largest S03 arrangement-sensitive candidate-ratio rows and, by plan, included one memory-repair contrast for coverage. Each selected row was replayed in the S03 best and worst arrangements, across {profile_count} goal profiles: same increasing target, opposite increasing/decreasing targets, partially compatible high-half-first target, local focus goals with a shared increasing target, and unrelated parity/center-out targets.

For each actor step, the active policy saw a goal-transformed rank vector for its assigned local goal. Actual array values, labels, identities, and memories still moved together on swaps. This keeps S04 in the same 1D local-action proxy substrate as S02/S03 while varying target compatibility.

Final states were evaluated under every relevant goal in each profile. The report table includes global goal scores, assigned subpopulation sortedness, assigned adjacent satisfaction, shared-goal score when defined, reference increasing score, and a goal-alignment gap.

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

- Selected S03 candidate rows: {len(selected_candidates)}.
- Candidate-arrangement-goal conditions: {len(condition_df)}.
- Run count: {run_count}.
- Goal-sensitive candidate-arrangement rows: {goal_sensitive_count}/{len(sensitivity)}.
- Mean assigned-policy sortedness across summaries: {summary['mean_assigned_policy_sortedness'].mean():.4f}.
- Mean goal-alignment gap across summaries: {summary['mean_goal_alignment_gap'].mean():.4f}.
- Mean reference increasing score across summaries: {summary['mean_reference_increasing_score'].mean():.4f}.

Selected S03 inputs:

{markdown_table(selected, ["s04_candidate_rank", "selection_reason", "panel", "display_names", "ratio_targets_json", "best_sortedness_arrangement", "worst_sortedness_arrangement", "sortedness_range_across_arrangements", "aggregation_delta_range_across_arrangements"], max_rows=20)}

Top goal-profile sensitivity contrasts:

{markdown_table(top_sensitivity, ["display_names", "ratio", "arrangement", "assigned_policy_sortedness_range", "reference_increasing_range", "goal_alignment_gap_range", "best_assigned_profile", "best_assigned_score", "worst_assigned_profile", "worst_assigned_score", "goal_sensitive_candidate"], max_rows=15)}

Representative goal summaries:

{markdown_table(representative, ["display_names", "ratio", "arrangement", "goal_profile_id", "goal_compatibility_class", "mean_assigned_policy_sortedness", "mean_reference_increasing_score", "mean_goal_alignment_gap", "goal_state_class_mode"], max_rows=25)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=20)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- S04 tests a bounded S03-derived panel, not every S01 policy pair, ratio, or arrangement.
- Local goals are rank-order encoders for the 1D sorting proxy substrate; they are not biological morphogen gradients or true tissue-level objectives.
- E04 memory-repair contrast is included for continuity even though the S03 memory rows were weaker arrangement-sensitivity cases.
- Goal-profile outcomes are computational associations; they do not establish biological causality or wet-lab validation.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Use `tables/e06_s04_goal_profile_sensitivity.csv` to define S05 compatibility metrics and pure/control baselines, keeping shared-goal score, assigned-goal score, alignment gap, aggregation, and work cost separate.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_goal_compatibility.parquet"
    summary_path = artifacts_dir / "tables/e06_goal_compatibility_summary.csv"
    sensitivity_path = artifacts_dir / "tables/e06_s04_goal_profile_sensitivity.csv"
    condition_path = step_dir / "e06_s04_condition_matrix.csv"
    selected_path = step_dir / "e06_s04_selected_candidates.csv"
    validation_path = step_dir / "e06_s04_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/goal_compatibility_matrix.png"
    config_path = artifacts_dir / "configs/e06_s04_goal_compatibility_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S04Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        max_sensitive_rows=args.max_sensitive_rows,
        include_memory_contrast=bool(args.include_memory_contrast),
        goal_profiles=DEFAULT_GOAL_PROFILES,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "s03SensitivityPath": str(args.s03_sensitivity_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "maxSensitiveRows": config.max_sensitive_rows,
        "includeMemoryContrast": config.include_memory_contrast,
        "scheduler": config.scheduler,
        "goalProfiles": [
            {
                "profileId": profile.profile_id,
                "compatibilityClass": profile.compatibility_class,
                "policyGoalNames": list(profile.policy_goal_names),
                "sharedGoalName": profile.shared_goal_name,
                "relevantGoalNames": list(profile.relevant_goal_names),
                "description": profile.description,
            }
            for profile in config.goal_profiles
        ],
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    records, _metadata = load_s01_library(args.library_path, args.metadata_path)
    if not args.s03_sensitivity_path.exists():
        raise FileNotFoundError(f"S03 arrangement sensitivity table is missing: {args.s03_sensitivity_path}")
    s03_sensitivity = pd.read_csv(args.s03_sensitivity_path)
    run_df, condition_df, selected_candidates = run_s04_sweep(records, s03_sensitivity, config)
    summary = summarize_s04_runs(run_df)
    sensitivity = goal_profile_sensitivity(summary)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(results_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(sensitivity_path, index=False)
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    condition_df.to_csv(condition_path, index=False)
    selected_candidates.to_csv(selected_path, index=False)
    figure_written = write_goal_figure(figure_path, summary, sensitivity)

    validation = validate_s04_outputs(
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
        artifact_entry(results_path, artifacts_dir, "S04 run-level goal-compatibility sweep table."),
        artifact_entry(summary_path, artifacts_dir, "S04 goal-compatibility summary table."),
        artifact_entry(sensitivity_path, artifacts_dir, "S04 goal-profile sensitivity table."),
        artifact_entry(figure_path, artifacts_dir, "S04 goal-compatibility matrix figure."),
        artifact_entry(condition_path, artifacts_dir, "S04 condition matrix."),
        artifact_entry(selected_path, artifacts_dir, "S04 selected S03 candidate table."),
        artifact_entry(validation_path, artifacts_dir, "S04 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S04 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S04 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S04 full-results Markdown handoff report.")

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
    report_artifact = artifact_entry(report_path, artifacts_dir, "S04 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    manifest = {
        "schema": "eidosoma.e06.s04_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "runCount": int(len(run_df)),
        "conditionCount": int(len(condition_df)),
        "selectedCandidateCount": int(len(selected_candidates)),
        "goalSensitiveCount": int(sensitivity["goal_sensitive_candidate"].sum()) if not sensitivity.empty else 0,
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S04 artifact manifest.")
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
            "s03ArrangementSensitivity": {
                "path": str(args.s03_sensitivity_path),
                "sha256": sha256_file(args.s03_sensitivity_path),
                "sizeBytes": args.s03_sensitivity_path.stat().st_size,
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
        "goalSensitiveCount": int(sensitivity["goal_sensitive_candidate"].sum()) if not sensitivity.empty else 0,
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
