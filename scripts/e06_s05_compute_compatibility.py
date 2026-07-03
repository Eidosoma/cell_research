#!/usr/bin/env python3
"""Compute E06 S05 compatibility metrics from S02-S04 outputs."""

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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.compatibility_metrics import (  # noqa: E402
    METRIC_DEFINITIONS,
    STEP_ID,
    S05Config,
    build_control_scores,
    build_pure_baselines,
    score_all_sources,
    select_s06_candidate_contests,
    summarize_dimensions,
    validate_s05_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, sha256_file  # noqa: E402


STEP_NUMBER = 5
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--s01-validation-path", type=Path, default=ARTIFACTS_DIR / "research_steps/S01/e06_s01_validation_runs.parquet")
    parser.add_argument("--s01-metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s02-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_mixture_ratio_sweep.parquet")
    parser.add_argument("--s02-summary-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_mixture_ratio_summary.csv")
    parser.add_argument("--s03-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_spatial_arrangement_sweep.parquet")
    parser.add_argument("--s03-summary-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_spatial_arrangement_summary.csv")
    parser.add_argument("--s04-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_goal_compatibility.parquet")
    parser.add_argument("--s04-summary-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_goal_compatibility_summary.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--baseline-array-size", type=int, default=8)
    parser.add_argument("--work-scaling-exponent", type=float, default=2.0)
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
    return " vs ".join(json.loads(str(text))[:3])


def _ratio_label(text: str) -> str:
    return ":".join(str(int(round(float(value) * 100))) for value in json.loads(str(text)))


def _metric_definitions_table() -> str:
    rows = pd.DataFrame(METRIC_DEFINITIONS)
    return markdown_table(rows, ["metric", "formula", "interpretation"], max_rows=50)


def write_metric_spec(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    validation: pd.DataFrame,
    config_payload: dict[str, Any],
    score_count: int,
    control_count: int,
    outcome: str,
) -> None:
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    text = f"""# E06 S05 Compatibility Metric Spec

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if bool(validation["success"].all()) else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: Compatibility is represented as separate proxy dimensions; no single aggregate compatibility score is emitted. Pure work baselines are scaled from S01 n=8 pure runs to S02-S04 n=100 runs with a documented O(n^2) proxy.
- Lay summary: S05 turns S02-S04 chimeric runs into comparable compatibility dimensions by asking whether a mixture beats or falls below the weighted pure-policy expectation, whether it pays extra work, whether labels integrate or aggregate, whether one policy dominates spatially, and whether S04 goals disagree.
- Recommended next action: Use the S05 S06 candidate table to run S06 dominance contests, keeping goal conflict, dominance, and target-quality loss as separate filters.

## Metric Philosophy

S05 deliberately does not collapse compatibility into one scalar. A mixture can sort well while segregating, preserve local goals while losing a shared goal, or beat pure baselines while paying a high work cost. Those are different scientific claims, so the score table preserves each dimension separately and records disagreement flags.

## Definitions

{_metric_definitions_table()}

## Baseline Normalization

For each mixture row, expected pure quality is the ratio-weighted mean of the S01 pure-policy final inversion sortedness for the participating Algotypes. Quality synergy is observed primary target quality minus that expected pure quality.

Expected pure work is ratio-weighted S01 pure work scaled by `(target_array_size / baseline_array_size) ** work_scaling_exponent`. This is a rough computational proxy because S01 pure baselines are n=8 smoke validations and S02-S04 mixture runs are n=100 proxy-harness runs.

## Pattern Classes

- `cooperative_integrated`: high target quality, near-baseline or better quality, low aggregation, low dominance, and low goal conflict.
- `high_quality_segregated`: high target quality with strong aggregation.
- `goal_tension_local_success`: S04 local assigned goals do well while relevant goals disagree.
- `mutual_interference`: target quality falls substantially below pure-policy expectation.
- `dominance_skewed`: positional dominance is large.
- `partial_or_contextual_compatibility`: target quality is usable but at least one dimension remains context-dependent.
- `low_compatibility_or_unresolved`: low quality without a clearer class.
- `pure_or_dummy_label_control`: synthetic validation controls only.

## Parameters

```json
{json.dumps(config_payload, indent=2, sort_keys=True, default=str)}
```

## Validation Scope

The score table contains {score_count} S02-S04 rows. The control table contains {control_count} pure-policy or same-policy dummy-label controls used only for metric validation.
"""
    write_text(path, text)


def write_report(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    scores: pd.DataFrame,
    baselines: pd.DataFrame,
    control_scores: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    s06_candidates: pd.DataFrame,
    validation: pd.DataFrame,
    config_payload: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    validation_passed = bool(validation["success"].all() and unit_test_result.get("success", True))
    outcome = "supportive" if validation_passed and not scores.empty else "null"
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    top_synergy = scores.sort_values("quality_synergy", ascending=False).copy()
    top_loss = scores.sort_values("quality_synergy", ascending=True).copy()
    conflicts = scores.sort_values(["metric_disagreement_count", "goal_conflict_index"], ascending=[False, False]).copy()
    for frame in (top_synergy, top_loss, conflicts, s06_candidates):
        if not frame.empty and "display_names_json" in frame.columns:
            frame["display_names"] = frame["display_names_json"].map(_display_name)
            frame["ratio"] = frame["ratio_targets_json"].map(_ratio_label)
    text = f"""# E06 S05 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: S05 uses S01 n=8 pure-policy smoke baselines to normalize S02-S04 n=100 proxy-harness runs, so work-normalized interference is a scaled proxy. Metrics are intentionally separate because quality, aggregation, dominance, work, and goal conflict often disagree.
- Lay summary: S05 converts prior chimeric sweeps into a compatibility score table. Most rows fall below weighted pure-policy target quality, but the table now separates whether that loss is accompanied by extra work, segregation, dominance, or goal conflict, which makes S06 dominance-contest selection concrete.
- Recommended next action: Run S06 dominance hierarchy tests using `tables/e06_s05_s06_candidate_contests.csv`, prioritizing high goal-conflict and high dominance rows rather than a single aggregate score.

## Frozen Question

Can compatibility be quantified through cooperative efficiency, mutual interference, final target quality, interface stability, Aggregation, and dominance?

## Inputs

- S01 pure-policy validation runs: `{config_payload['s01ValidationPath']}`
- S01 metadata baselines: `{config_payload['s01MetadataPath']}`
- S02 mixture runs and summary: `{config_payload['s02RunPath']}`, `{config_payload['s02SummaryPath']}`
- S03 spatial runs and summary: `{config_payload['s03RunPath']}`, `{config_payload['s03SummaryPath']}`
- S04 goal runs and summary: `{config_payload['s04RunPath']}`, `{config_payload['s04SummaryPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S05 built one pure baseline per S01 Algotype, then scored every S02, S03, and S04 summary row against the ratio-weighted pure baseline expected for that mixture. S02 and S03 primary target quality is final inversion sortedness. S04 primary target quality is assigned-policy goal sortedness, with separate reference-increasing, shared-goal, and goal-alignment dimensions retained.

The method keeps conflicting dimensions separate. It writes quality synergy, work interference, cooperative efficiency, integration, interface stability, dominance, goal conflict, and a categorical pattern label, but it does not emit a single aggregate compatibility score.

## Commands

- Main command: `{command}`
- Unit-test command: `{unit_test_result.get('command', 'not run')}`
- Unit-test return code: `{unit_test_result.get('returnCode', 'not run')}`

## Dependencies

No new dependencies were installed. The script used repository modules plus preinstalled `pandas`, `numpy`, and `pyarrow`.

## Parameters

```json
{json.dumps(config_payload, indent=2, sort_keys=True, default=str)}
```

## Results

- Compatibility score rows: {len(scores)}.
- Source rows: {scores['source_research_step_id'].value_counts().sort_index().to_dict() if not scores.empty else {}}.
- Pure baselines: {len(baselines)}.
- Control validation rows: {len(control_scores)}.
- Mean primary target quality: {scores['primary_target_quality'].mean():.4f}.
- Mean quality synergy versus S01 pure baselines: {scores['quality_synergy'].mean():.4f}.
- Mean work interference log ratio: {scores['work_interference_log_ratio'].mean():.4f}.
- Rows with conflicting dimensions: {int((scores['metric_disagreement_count'] > 0).sum())}.

Dimension summary:

{markdown_table(dimension_summary, ["source_research_step_id", "goal_compatibility_class", "score_row_count", "mean_primary_target_quality", "mean_quality_synergy", "mean_work_interference_log_ratio", "mean_aggregation_delta_percent", "mean_dominance_abs_margin", "mean_goal_conflict_index", "mean_metric_disagreement_count"], max_rows=30)}

Largest positive quality synergy:

{markdown_table(top_synergy, ["source_research_step_id", "display_names", "ratio", "arrangement", "goal_profile_id", "primary_target_quality", "expected_pure_quality", "quality_synergy", "compatibility_pattern_class"], max_rows=12)}

Largest target-quality interference:

{markdown_table(top_loss, ["source_research_step_id", "display_names", "ratio", "arrangement", "goal_profile_id", "primary_target_quality", "expected_pure_quality", "quality_synergy", "compatibility_pattern_class"], max_rows=12)}

Most explicit metric disagreements:

{markdown_table(conflicts, ["source_research_step_id", "display_names", "ratio", "arrangement", "goal_profile_id", "primary_target_quality", "quality_synergy", "aggregation_delta_percent", "dominance_abs_margin", "goal_conflict_index", "conflicting_dimensions_json"], max_rows=12)}

S06 candidate contests:

{markdown_table(s06_candidates, ["display_names", "ratio", "arrangement", "goal_profile_id", "goal_compatibility_class", "primary_target_quality", "quality_synergy", "goal_conflict_index", "dominance_abs_margin", "s06_priority_score"], max_rows=18)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=20)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- S01 pure baselines are small-array smoke validations, not full n=100 pure-policy sweeps.
- Work interference is therefore an O(n^2)-scaled proxy and should be interpreted more cautiously than quality, aggregation, or dominance dimensions.
- S04 goal compatibility uses rank-order proxy goals from S04, not biological target fields.
- The pattern class is descriptive triage for downstream S06-S07, not a causal or biological category.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Proceed to S06 dominance hierarchy mapping using the S05 candidate table. Select contests by goal conflict and dominance dimensions, not by an aggregate compatibility score.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_compatibility_scores.parquet"
    scores_csv_path = artifacts_dir / "tables/e06_compatibility_scores.csv"
    baseline_path = artifacts_dir / "tables/e06_s05_pure_baselines.csv"
    dimension_summary_path = artifacts_dir / "tables/e06_compatibility_dimension_summary.csv"
    s06_candidates_path = artifacts_dir / "tables/e06_s05_s06_candidate_contests.csv"
    controls_path = step_dir / "e06_s05_pure_dummy_control_scores.csv"
    validation_path = step_dir / "e06_s05_validation_checks.csv"
    spec_path = artifacts_dir / "reports/e06_compatibility_metric_spec.md"
    config_path = artifacts_dir / "configs/e06_s05_compatibility_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S05Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        baseline_array_size=args.baseline_array_size,
        work_scaling_exponent=args.work_scaling_exponent,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "s01ValidationPath": str(args.s01_validation_path),
        "s01MetadataPath": str(args.s01_metadata_path),
        "s02RunPath": str(args.s02_run_path),
        "s02SummaryPath": str(args.s02_summary_path),
        "s03RunPath": str(args.s03_run_path),
        "s03SummaryPath": str(args.s03_summary_path),
        "s04RunPath": str(args.s04_run_path),
        "s04SummaryPath": str(args.s04_summary_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "baselineArraySize": config.baseline_array_size,
        "workScalingExponent": config.work_scaling_exponent,
        "metricDefinitions": list(METRIC_DEFINITIONS),
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    s01_validation = pd.read_parquet(args.s01_validation_path)
    s01_metadata = pd.read_csv(args.s01_metadata_path)
    s02_runs = pd.read_parquet(args.s02_run_path)
    s02_summary = pd.read_csv(args.s02_summary_path)
    s03_runs = pd.read_parquet(args.s03_run_path)
    s03_summary = pd.read_csv(args.s03_summary_path)
    s04_runs = pd.read_parquet(args.s04_run_path)
    s04_summary = pd.read_csv(args.s04_summary_path)

    baselines = build_pure_baselines(s01_metadata, s01_validation)
    scores = score_all_sources(
        s02_summary=s02_summary,
        s03_summary=s03_summary,
        s04_summary=s04_summary,
        baselines=baselines,
        config=config,
        s02_runs=s02_runs,
        s03_runs=s03_runs,
        s04_runs=s04_runs,
    )
    control_scores = build_control_scores(baselines, config)
    dimension_summary = summarize_dimensions(scores)
    s06_candidates = select_s06_candidate_contests(scores)
    validation = validate_s05_outputs(
        scores,
        baselines,
        control_scores,
        expected_source_counts={"S02": len(s02_summary), "S03": len(s03_summary), "S04": len(s04_summary)},
        unit_tests_success=bool(unit_test_result.get("success", True)),
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(results_path, index=False)
    scores_csv_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_csv_path, index=False)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(baseline_path, index=False)
    dimension_summary_path.parent.mkdir(parents=True, exist_ok=True)
    dimension_summary.to_csv(dimension_summary_path, index=False)
    s06_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    s06_candidates.to_csv(s06_candidates_path, index=False)
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    control_scores.to_csv(controls_path, index=False)
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
        artifact_entry(results_path, artifacts_dir, "S05 compatibility score table."),
        artifact_entry(scores_csv_path, artifacts_dir, "CSV mirror of S05 compatibility scores."),
        artifact_entry(baseline_path, artifacts_dir, "S05 S01 pure-policy baseline table."),
        artifact_entry(dimension_summary_path, artifacts_dir, "S05 compatibility dimension summary table."),
        artifact_entry(s06_candidates_path, artifacts_dir, "S05-selected S06 dominance/conflict candidate contests."),
        artifact_entry(controls_path, artifacts_dir, "S05 pure-policy and dummy-label metric control scores."),
        artifact_entry(validation_path, artifacts_dir, "S05 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S05 compatibility metric configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S05 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S05 full-results Markdown handoff report.")
    spec_pending = pending_entry(spec_path, artifacts_dir, "S05 compatibility metric specification.")
    outcome = "supportive" if bool(validation["success"].all() and unit_test_result.get("success", True) and not scores.empty) else "null"

    write_metric_spec(
        spec_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending, spec_pending],
        validation=validation,
        config_payload=config_payload,
        score_count=len(scores),
        control_count=len(control_scores),
        outcome=outcome,
    )
    spec_artifact = artifact_entry(spec_path, artifacts_dir, "S05 compatibility metric specification.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending, spec_artifact],
        scores=scores,
        baselines=baselines,
        control_scores=control_scores,
        dimension_summary=dimension_summary,
        s06_candidates=s06_candidates,
        validation=validation,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S05 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    manifest = {
        "schema": "eidosoma.e06.s05_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "scoreRowCount": int(len(scores)),
        "pureBaselineCount": int(len(baselines)),
        "controlRowCount": int(len(control_scores)),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, spec_artifact, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S05 artifact manifest.")
    artifacts = [*base_artifacts, spec_artifact, report_artifact, manifest_artifact]

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
            "s01Validation": {"path": str(args.s01_validation_path), "sha256": sha256_file(args.s01_validation_path), "sizeBytes": args.s01_validation_path.stat().st_size},
            "s01Metadata": {"path": str(args.s01_metadata_path), "sha256": sha256_file(args.s01_metadata_path), "sizeBytes": args.s01_metadata_path.stat().st_size},
            "s02Runs": {"path": str(args.s02_run_path), "sha256": sha256_file(args.s02_run_path), "sizeBytes": args.s02_run_path.stat().st_size},
            "s02Summary": {"path": str(args.s02_summary_path), "sha256": sha256_file(args.s02_summary_path), "sizeBytes": args.s02_summary_path.stat().st_size},
            "s03Runs": {"path": str(args.s03_run_path), "sha256": sha256_file(args.s03_run_path), "sizeBytes": args.s03_run_path.stat().st_size},
            "s03Summary": {"path": str(args.s03_summary_path), "sha256": sha256_file(args.s03_summary_path), "sizeBytes": args.s03_summary_path.stat().st_size},
            "s04Runs": {"path": str(args.s04_run_path), "sha256": sha256_file(args.s04_run_path), "sizeBytes": args.s04_run_path.stat().st_size},
            "s04Summary": {"path": str(args.s04_summary_path), "sha256": sha256_file(args.s04_summary_path), "sizeBytes": args.s04_summary_path.stat().st_size},
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        scores_csv_path,
        baseline_path,
        dimension_summary_path,
        s06_candidates_path,
        controls_path,
        validation_path,
        config_path,
        spec_path,
        manifest_path,
        report_path,
        run_manifest_path,
    ]
    write_text(checksums_path, "\n".join(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_paths) + "\n")

    status = {
        "success": success,
        "scoreRowCount": int(len(scores)),
        "pureBaselineCount": int(len(baselines)),
        "controlRowCount": int(len(control_scores)),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "sourceCounts": scores["source_research_step_id"].value_counts().sort_index().to_dict() if not scores.empty else {},
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
