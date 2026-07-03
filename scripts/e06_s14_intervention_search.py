#!/usr/bin/env python3
"""Run E06 S14 prospective minimal intervention search."""

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

from src.e06.intervention_search import (  # noqa: E402
    BASELINE_SPEC_ID,
    SEARCH_STAGE,
    STEP_ID,
    VALIDATION_STAGE,
    S14Config,
    compare_to_baseline,
    outcome_classification,
    rank_interventions,
    run_s14_search,
    validate_s14_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402


STEP_NUMBER = 14
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s13-matrix-path", type=Path, default=ARTIFACTS_DIR / "research_steps/S13/e06_s13_model_input_matrix.parquet")
    parser.add_argument("--s13-prediction-path", type=Path, default=ARTIFACTS_DIR / "results/e06_final_state_prediction.parquet")
    parser.add_argument("--s13-feature-screen-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s13_exploratory_causal_screen.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-search-contexts", type=int, default=4)
    parser.add_argument("--max-holdout-contexts", type=int, default=2)
    parser.add_argument("--search-seeds", type=int, nargs="*", default=[2026071401, 2026071402])
    parser.add_argument("--validation-seeds", type=int, nargs="*", default=[2026071491, 2026071492, 2026071493, 2026071494])
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


def source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "sizeBytes": path.stat().st_size if path.exists() else None,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
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


def write_intervention_figure(path: Path, comparison: pd.DataFrame, rankings: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if comparison.empty or rankings.empty:
        return False
    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_min = fig.add_subplot(grid[0, 1])
    ax_unintended = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])

    validation = rankings[(rankings["evaluation_stage"] == VALIDATION_STAGE) & (rankings["intervention_spec_id"] != BASELINE_SPEC_ID)].copy()
    if validation.empty:
        validation = rankings[rankings["intervention_spec_id"] != BASELINE_SPEC_ID].copy()
    validation = validation.sort_values("mean_rescue_effect_score", ascending=False, kind="mergesort")
    colors = np.where(validation["global_controller_like"].astype(bool), "#8c3f97", "#2f7f7f")
    ax_rank.bar(validation["intervention_spec_id"], validation["mean_rescue_effect_score"], color=colors)
    ax_rank.axhline(0, color="#222222", lw=0.8)
    ax_rank.set_title("Prospective rescue score by intervention")
    ax_rank.set_ylabel("Mean rescue-effect score")
    ax_rank.tick_params(axis="x", rotation=45, labelsize=8)

    plot = rankings[rankings["intervention_spec_id"] != BASELINE_SPEC_ID].copy()
    for global_like, group in plot.groupby("global_controller_like", sort=False):
        ax_min.scatter(
            group["minimality_score"],
            group["mean_rescue_effect_score"],
            s=70,
            alpha=0.75,
            label="global-controller-like" if global_like else "local or finite-radius",
        )
    ax_min.axhline(0, color="#222222", lw=0.8, linestyle="--")
    ax_min.set_title("Minimality and rescue")
    ax_min.set_xlabel("Minimality score; lower is simpler")
    ax_min.set_ylabel("Mean rescue-effect score")
    ax_min.legend(fontsize=8)

    ax_unintended.bar(validation["intervention_spec_id"], validation["mean_unintended_consequence_score"], color=colors)
    ax_unintended.set_title("Unintended-consequence score")
    ax_unintended.set_ylabel("Mean score")
    ax_unintended.tick_params(axis="x", rotation=45, labelsize=8)

    heldout = comparison[(comparison["evaluation_stage"] == VALIDATION_STAGE) & (comparison["intervention_spec_id"] != BASELINE_SPEC_ID)].copy()
    if heldout.empty:
        heldout = comparison[comparison["intervention_spec_id"] != BASELINE_SPEC_ID].copy()
    if not heldout.empty:
        pivot = heldout.pivot_table(
            index="base_context_id",
            columns="intervention_spec_id",
            values="rescue_effect_score",
            aggfunc="mean",
        ).fillna(0.0)
        matrix = pivot.to_numpy(dtype=float)
        max_abs = max(0.001, float(np.nanmax(np.abs(matrix))))
        image = ax_heat.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax_heat.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right", fontsize=8)
        ax_heat.set_yticks(np.arange(len(pivot.index)), labels=pivot.index, fontsize=8)
        ax_heat.set_title("Held-out context rescue heatmap")
        fig.colorbar(image, ax=ax_heat, shrink=0.85)
    else:
        ax_heat.text(0.5, 0.5, "No held-out comparison rows", ha="center", va="center")
        ax_heat.set_axis_off()

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def write_report(
    path: Path,
    *,
    artifacts_written: list[str],
    config: S14Config,
    input_paths: dict[str, Path],
    selected: pd.DataFrame,
    specs: pd.DataFrame,
    run_df: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    rankings: pd.DataFrame,
    validation: pd.DataFrame,
    outcome: str,
    unit_test_result: dict[str, Any],
    command_line: str,
) -> None:
    validation_passed = bool(validation["success"].all()) if not validation.empty else False
    source_counts = selected["source_research_step_id"].value_counts().sort_index().reset_index()
    source_counts.columns = ["source_research_step_id", "selected_context_count"]
    top_validation = rankings[rankings["evaluation_stage"] == VALIDATION_STAGE].sort_values(
        ["global_controller_like", "mean_minimality_adjusted_rescue_score", "minimality_score"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    report = f"""# E06 S14 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{chr(10).join(f'- `{item}`' for item in artifacts_written)}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and focused unit tests {"passed" if unit_test_result.get("success") else "failed"}.
- Outcome classification: {outcome}
- Caveats or blockers: S14 remains a computational sorting-array intervention search. S13 feature rankings were used only as design heuristics; they are not causal proof. Explicit local-recognition/interface interventions and global-controller-like comparators are labeled separately from original no-recognition behavior-only claims.
- Lay summary: S14 picked high-failure contexts from S13, searched short transient local signals and local rule changes on new seeds, then validated the best local candidates plus a global comparator on separate held-out contexts and held-out seeds. Each result is compared to a matched no-intervention baseline.
- Recommended next action: Proceed to S15 playbook synthesis using S14's held-out intervention rankings, minimality scores, and unintended-consequence metrics; keep global-controller-like evidence separate from local intervention evidence.

## Frozen Question

Can minimal transient signals or local rule changes rescue failed chimeras or steer them to desired equilibria?

## Inputs

{markdown_table(pd.DataFrame([{"input_name": key, "path": str(value), "exists": value.exists()} for key, value in input_paths.items()]), ["input_name", "path", "exists"])}

## Methods

S14 used the S13 model input matrix, held-out predictions, and exploratory feature screen to select bounded pairwise failure contexts. The context score emphasized low final target quality, high goal conflict, S13 random-forest classification errors, quality residuals, and failure-like final-state classes. Search and validation contexts were selected as disjoint `base_context_id` sets.

Candidate interventions were predeclared as minimal transient pulses or local rule changes: behavior-only baseline, short early local interface pulses, short finite-radius quorum/conflict/organizer signals, one full-run local-recognition rule-change comparator, and one explicitly global-reference pulse comparator. Most intervention specs are active for 15% of the event horizon; outside that window, runs revert to behavior-only and no governance. The global comparator is retained only to mark what stronger nonlocal access could do.

The search stage ran all intervention specs over S13 failure contexts using new seeds. The prospective validation stage used separate held-out contexts and held-out seeds, running the no-intervention baseline, the top minimal local candidates from search, and the top global-controller-like comparator. Comparisons are always made against the matched baseline for the same stage and context.

Minimality is quantified as active duration plus access cost and finite-radius cost. Unintended consequences are reported as a separate score combining target-quality loss, conflict increase, work increase, block fractions, and global-controller-like access penalty. Rescue-effect scores do not override those separate dimensions.

## Commands

- Main command: `{command_line}`
- Unit-test command: `{unit_test_result.get("command", "not run")}`
- Unit-test return code: `{unit_test_result.get("returnCode", "not run")}`

## Dependencies

No new dependencies were installed. S14 used repository code plus preinstalled `pandas`, `numpy`, `pyarrow`, and `matplotlib`.

## Parameters

```json
{json.dumps({
    "arraySize": config.array_size,
    "eventCap": config.event_cap,
    "searchSeeds": list(config.search_seeds),
    "validationSeeds": list(config.validation_seeds),
    "maxSearchContexts": config.max_search_contexts,
    "maxHoldoutContexts": config.max_holdout_contexts,
    "validationLocalInterventionCount": config.validation_local_intervention_count,
    "workerCount": config.worker_count,
    "rescueDeltaThreshold": config.rescue_delta_threshold,
    "minimalityPenaltyWeight": config.minimality_penalty_weight,
}, indent=2)}
```

## Results

- Run rows: {len(run_df)}
- Selected failure contexts: {len(selected)}
- Search contexts: {int((selected["evaluation_stage"] == SEARCH_STAGE).sum()) if not selected.empty else 0}
- Held-out validation contexts: {int((selected["evaluation_stage"] == VALIDATION_STAGE).sum()) if not selected.empty else 0}
- Intervention specs: {len(specs)}
- Summary rows: {len(summary)}
- Baseline-comparison rows: {len(comparison)}

### Selected Context Sources

{markdown_table(source_counts, ["source_research_step_id", "selected_context_count"])}

### Intervention Specs

{markdown_table(specs.sort_values("minimality_score"), ["intervention_spec_id", "mechanism_family", "schedule_family", "duration_fraction", "access_stratum", "global_controller_like", "minimality_score", "intended_effect"], max_rows=20)}

### Prospective Validation Rankings

{markdown_table(top_validation, ["intervention_spec_id", "access_stratum", "global_controller_like", "condition_count", "mean_delta_final_target_quality", "mean_conflict_reduction", "mean_rescue_effect_score", "mean_minimality_adjusted_rescue_score", "mean_unintended_consequence_score", "target_quality_rescue_rate", "supportive_rescue_by_threshold"], max_rows=20)}

### Search Rankings

{markdown_table(rankings[rankings["evaluation_stage"] == SEARCH_STAGE].sort_values("mean_minimality_adjusted_rescue_score", ascending=False, kind="mergesort"), ["intervention_spec_id", "access_stratum", "global_controller_like", "condition_count", "mean_delta_final_target_quality", "mean_conflict_reduction", "mean_rescue_effect_score", "mean_minimality_adjusted_rescue_score", "mean_unintended_consequence_score"], max_rows=20)}

## Validation

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=40)}

## Artifacts

{markdown_table(pd.DataFrame({"relative_path": artifacts_written}), ["relative_path"], max_rows=80)}

## Provenance

- Experiment ID: {EXPERIMENT_ID}
- Research step number: {STEP_NUMBER}
- Script: `scripts/e06_s14_intervention_search.py`
- Repository branch: `{subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True).stdout.strip()}`
- Repository commit: `{subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True).stdout.strip()}`
- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`

## Caveats And Limitations

- S14 tests computational proxies in 1D sorting arrays; it is not biological validation of transient signals, adhesion, or governance.
- S13 model features were used to choose intervention families and failure contexts, not to claim causal mechanisms.
- Held-out validation contexts are disjoint from search contexts, but all are drawn from the E06 proxy corpus and remain bounded to pairwise policies present in the S01 library.
- Global-controller-like results are explicitly separated and should not be counted as local chimeric governance evidence.
- Full-run local recognition is included as a heavier comparator; it is less minimal than transient pulses.

## Failed Assumptions Or Blockers

No blocking inputs were missing. The main bounded assumption is that S14 excludes S11 mutant pseudo-policy rows whose generated clone labels are not executable S01 library policies.
"""
    write_text(path, report)


def main() -> int:
    args = parse_args()
    started = utc_now()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps/S14"
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures/e06"
    configs_dir = artifacts_dir / "configs"
    checksums_dir = artifacts_dir / "checksums"

    config = S14Config(
        array_size=int(args.array_size),
        event_cap=int(args.event_cap),
        search_seeds=tuple(int(seed) for seed in args.search_seeds),
        validation_seeds=tuple(int(seed) for seed in args.validation_seeds),
        max_search_contexts=int(args.max_search_contexts),
        max_holdout_contexts=int(args.max_holdout_contexts),
        worker_count=int(args.workers),
    )

    records, metadata = load_s01_library(args.library_path, args.metadata_path)
    s13_matrix = pd.read_parquet(args.s13_matrix_path)
    predictions = pd.read_parquet(args.s13_prediction_path)
    feature_screen = pd.read_csv(args.s13_feature_screen_path)

    run_df, condition_df, selected, specs, summary, comparison, rankings = run_s14_search(
        records,
        s13_matrix,
        predictions,
        feature_screen,
        config,
    )

    run_path = results_dir / "e06_intervention_search.parquet"
    condition_path = step_dir / "e06_s14_condition_matrix.csv"
    selected_path = step_dir / "e06_s14_selected_failure_contexts.csv"
    specs_path = step_dir / "e06_s14_intervention_specs.csv"
    summary_csv = tables_dir / "e06_s14_intervention_summary.csv"
    summary_parquet = tables_dir / "e06_s14_intervention_summary.parquet"
    comparison_csv = tables_dir / "e06_s14_baseline_comparison.csv"
    rankings_csv = tables_dir / "e06_s14_intervention_rankings.csv"
    config_path = configs_dir / "e06_s14_intervention_search_config.json"
    validation_path = step_dir / "e06_s14_validation_checks.csv"
    figure_path = figures_dir / "intervention_rescue_examples.png"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(run_path, index=False)
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    condition_df.to_csv(condition_path, index=False)
    selected.to_csv(selected_path, index=False)
    specs.to_csv(specs_path, index=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_parquet, index=False)
    comparison.to_csv(comparison_csv, index=False)
    rankings.to_csv(rankings_csv, index=False)
    write_json(
        config_path,
        {
            "schema": "eidosoma.e06.s14_config.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "arraySize": config.array_size,
            "eventCap": config.event_cap,
            "searchSeeds": list(config.search_seeds),
            "validationSeeds": list(config.validation_seeds),
            "maxSearchContexts": config.max_search_contexts,
            "maxHoldoutContexts": config.max_holdout_contexts,
            "validationLocalInterventionCount": config.validation_local_intervention_count,
            "workerCount": config.worker_count,
            "interventionSpecIds": specs["intervention_spec_id"].astype(str).tolist(),
        },
    )

    figure_written = write_intervention_figure(figure_path, comparison, rankings)
    unit_test_result = (
        run_command([sys.executable, "-m", "unittest", "tests.e06.test_intervention_search", "-v"], cwd=args.repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": None, "success": True, "stdout": "", "stderr": ""}
    )
    validation = validate_s14_outputs(
        run_df,
        condition_df,
        selected,
        specs,
        summary,
        comparison,
        rankings,
        config,
        figure_written=figure_written,
        unit_tests_success=bool(unit_test_result.get("success")),
    )
    validation.to_csv(validation_path, index=False)
    validation_passed = bool(validation["success"].all())
    outcome = outcome_classification(rankings, validation_passed)

    artifact_specs = [
        (run_path, "S14 run-level intervention search outcomes"),
        (condition_path, "S14 executable condition matrix"),
        (selected_path, "S14 S13-derived failure context selection"),
        (specs_path, "S14 intervention definitions and minimality metadata"),
        (summary_csv, "S14 intervention summary table"),
        (summary_parquet, "S14 intervention summary table in Parquet"),
        (comparison_csv, "S14 matched baseline comparison with unintended consequences"),
        (rankings_csv, "S14 search and prospective validation rankings"),
        (figure_path, "S14 intervention rescue examples figure"),
        (config_path, "S14 run configuration"),
        (validation_path, "S14 validation checks"),
        (report_path, "S14 full-results report"),
        (manifest_path, "S14 artifact manifest"),
        (run_manifest_path, "Experiment run manifest"),
        (checksum_path, "SHA256 checksum manifest"),
    ]
    artifact_rel = [str(path.relative_to(artifacts_dir)) for path, _ in artifact_specs]
    write_report(
        report_path,
        artifacts_written=artifact_rel,
        config=config,
        input_paths={
            "s01_library": args.library_path,
            "s01_metadata": args.metadata_path,
            "s13_model_input_matrix": args.s13_matrix_path,
            "s13_heldout_predictions": args.s13_prediction_path,
            "s13_feature_screen": args.s13_feature_screen_path,
        },
        selected=selected,
        specs=specs,
        run_df=run_df,
        summary=summary,
        comparison=comparison,
        rankings=rankings,
        validation=validation,
        outcome=outcome,
        unit_test_result=unit_test_result,
        command_line=" ".join(sys.argv),
    )

    manifest = {
        "schema": "eidosoma.e06.s14_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "startedAt": started,
        "success": bool(validation_passed),
        "status": "complete" if validation_passed else "validation_failed",
        "outcomeClassification": outcome,
        "rowCount": int(len(run_df)),
        "selectedContextCount": int(len(selected)),
        "searchContextCount": int((selected["evaluation_stage"] == SEARCH_STAGE).sum()),
        "heldoutContextCount": int((selected["evaluation_stage"] == VALIDATION_STAGE).sum()),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [artifact_entry(path, artifacts_dir, description) for path, description in artifact_specs if path.exists()],
        "sources": {
            "library": source_entry(args.library_path),
            "metadata": source_entry(args.metadata_path),
            "s13Matrix": source_entry(args.s13_matrix_path),
            "s13Predictions": source_entry(args.s13_prediction_path),
            "s13FeatureScreen": source_entry(args.s13_feature_screen_path),
        },
        "unitTests": unit_test_result,
        "repoState": {
            "branch": git_output(args.repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
    }
    write_json(manifest_path, manifest)

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repoBranch": git_output(args.repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "repoHead": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "outcomeClassification": outcome,
        "artifactManifest": str(manifest_path),
    }
    write_json(run_manifest_path, run_manifest)

    checksum_paths = [path for path, _ in artifact_specs if path.exists() and path != checksum_path]
    checksums_dir.mkdir(parents=True, exist_ok=True)
    checksum_lines = []
    for path in checksum_paths:
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksum_path, "\n".join(checksum_lines) + "\n")

    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": validation_passed,
                "status": "complete" if validation_passed else "validation_failed",
                "outcomeClassification": outcome,
                "rowCount": int(len(run_df)),
                "validationPassed": int(validation["success"].sum()),
                "validationTotal": int(len(validation)),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
