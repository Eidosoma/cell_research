#!/usr/bin/env python3
"""Run E06 S07 mosaic and final-state formation analysis."""

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
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.mosaic_formation import (  # noqa: E402
    CLASSIFIER_FEATURE_COLUMNS,
    STEP_ID,
    S07Config,
    add_continuum_coordinates,
    add_rule_labels,
    attach_explanatory_contexts,
    classification_mode,
    condition_seed_stability,
    harmonize_all_runs,
    metric_continua_summary,
    run_classifier_settings,
    select_exemplars,
    summarize_classes,
    validate_s07_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, sha256_file  # noqa: E402


STEP_NUMBER = 7
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--s02-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_mixture_ratio_sweep.parquet")
    parser.add_argument("--s03-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_spatial_arrangement_sweep.parquet")
    parser.add_argument("--s04-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_goal_compatibility.parquet")
    parser.add_argument("--s05-score-path", type=Path, default=ARTIFACTS_DIR / "results/e06_compatibility_scores.parquet")
    parser.add_argument("--s06-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_dominance_contests.parquet")
    parser.add_argument("--s06-hierarchy-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_dominance_hierarchy.csv")
    parser.add_argument("--s06-context-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s06_context_dependence.csv")
    parser.add_argument("--seed-stability-threshold", type=float, default=0.75)
    parser.add_argument("--classifier-stability-threshold", type=float, default=0.70)
    parser.add_argument("--classifier-min-pairwise-agreement", type=float, default=0.55)
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


def _json_loads(text: Any) -> list[Any]:
    try:
        return list(json.loads(str(text)))
    except json.JSONDecodeError:
        return []


def _display_name(text: str) -> str:
    try:
        return " vs ".join(str(item) for item in json.loads(str(text))[:3])
    except json.JSONDecodeError:
        return str(text)


def _short_label(label: str, limit: int = 28) -> str:
    label = str(label)
    return label if len(label) <= limit else label[: limit - 1] + "."


def _palette(values: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {value: cmap(idx % 20) for idx, value in enumerate(values)}


def write_mosaic_figure(path: Path, mosaic: pd.DataFrame, exemplars: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mosaic.empty:
        return False
    labels = sorted(mosaic["s07_label"].astype(str).unique())
    colors = _palette(labels)
    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.2])
    ax_pc = fig.add_subplot(grid[0, 0])
    ax_metric = fig.add_subplot(grid[0, 1])
    ax_conflict = fig.add_subplot(grid[1, 0])
    ax_source = fig.add_subplot(grid[1, 1])
    ax_strip = fig.add_subplot(grid[2, :])

    for label in labels:
        subset = mosaic[mosaic["s07_label"].astype(str) == label]
        ax_pc.scatter(
            subset["s07_continuum_pc1"],
            subset["s07_continuum_pc2"],
            s=18,
            alpha=0.65,
            color=colors[label],
            label=_short_label(label),
        )
        ax_metric.scatter(
            subset["aggregation_delta_percent"],
            subset["final_target_quality"],
            s=18,
            alpha=0.65,
            color=colors[label],
        )
        ax_conflict.scatter(
            subset["goal_conflict_index"],
            subset["position_bias_abs"],
            s=18,
            alpha=0.65,
            color=colors[label],
        )
    ax_pc.set_title("Continuum embedding")
    ax_pc.set_xlabel("S07 PC1")
    ax_pc.set_ylabel("S07 PC2")
    ax_metric.set_title("Target quality vs aggregation")
    ax_metric.set_xlabel("Aggregation delta percent")
    ax_metric.set_ylabel("Final target quality")
    ax_conflict.set_title("Goal conflict vs position bias")
    ax_conflict.set_xlabel("Goal conflict index")
    ax_conflict.set_ylabel("Position-bias magnitude")
    ax_pc.legend(loc="best", fontsize=7, ncols=2)

    source_counts = mosaic.groupby(["source_research_step_id", "s07_class_family"]).size().unstack(fill_value=0)
    if not source_counts.empty:
        source_counts.plot(kind="bar", stacked=True, ax=ax_source, colormap="tab20")
    ax_source.set_title("Class families by source step")
    ax_source.set_xlabel("Source step")
    ax_source.set_ylabel("Run count")
    ax_source.legend(loc="best", fontsize=7)

    ax_strip.set_title("Representative final-label strips (first 20 cells | last 20 cells)")
    ax_strip.axis("off")
    if not exemplars.empty:
        policy_ids = sorted({str(item) for text in pd.concat([exemplars["final_labels_head_json"], exemplars["final_labels_tail_json"]]) for item in _json_loads(text)})
        policy_colors = _palette(policy_ids)
        max_rows = min(len(exemplars), 12)
        for row_idx, row in enumerate(exemplars.head(max_rows).to_dict(orient="records")):
            head = [str(item) for item in _json_loads(row["final_labels_head_json"])]
            tail = [str(item) for item in _json_loads(row["final_labels_tail_json"])]
            labels_for_row = head[:20] + ["__gap__"] + tail[-20:]
            y = max_rows - row_idx - 1
            ax_strip.text(-2.0, y + 0.35, _short_label(row["s07_label"], 24), fontsize=8, ha="right", va="center")
            for col_idx, policy_id in enumerate(labels_for_row):
                if policy_id == "__gap__":
                    ax_strip.add_patch(plt.Rectangle((col_idx, y), 0.3, 0.7, color="white", ec="black", lw=0.3))
                    continue
                ax_strip.add_patch(plt.Rectangle((col_idx, y), 0.85, 0.7, color=policy_colors.get(policy_id, "gray"), ec="none"))
            ax_strip.text(len(labels_for_row) + 0.8, y + 0.35, f"{row['source_research_step_id']} {row['ratio_label']}", fontsize=7, va="center")
        ax_strip.set_xlim(-9, 50)
        ax_strip.set_ylim(-0.5, max_rows + 0.5)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def write_report(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    mosaic: pd.DataFrame,
    class_summary: pd.DataFrame,
    continua: pd.DataFrame,
    condition_stability: pd.DataFrame,
    classifier_settings: pd.DataFrame,
    classifier_pairwise: pd.DataFrame,
    exemplars: pd.DataFrame,
    validation: pd.DataFrame,
    config_payload: dict[str, Any],
    stability_context: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    validation_passed = bool(validation["success"].all() and unit_test_result.get("success", True))
    mode = str(stability_context["classification_mode"])
    outcome = "supportive" if validation_passed and mode == "stable_discrete_taxonomy" else ("constraining/contradictory" if validation_passed else "null")
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    class_display = class_summary.copy()
    continua_display = continua[continua["metric"].isin(["final_target_quality", "goal_conflict_index", "aggregation_delta_percent", "largest_block_fraction", "position_bias_abs", "mosaic_continuum_score", "conflict_continuum_score"])].copy()
    exemplar_display = exemplars.copy()
    if not exemplar_display.empty:
        exemplar_display["display_names"] = exemplar_display["display_names_json"].map(_display_name)
    unstable = condition_stability.sort_values("majority_fraction", ascending=True).head(15).copy()
    text = f"""# E06 S07 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: S07 found that discrete labels are only a coarse overlay on continuous final-state metrics. The analysis has final event-cap states and trajectory proxies, but not full time-series traces, so oscillation and long-run turnover cannot be directly verified from S02-S06.
- Lay summary: S07 pooled the prior chimeric runs and described final states along target quality, aggregation, interface, block size, dominance-bias, activity, and goal-conflict continua. Because seed and classifier-setting stability did not both support a single crisp taxonomy, the handoff preserves metric continua and exemplar panels rather than forcing unstable class boundaries.
- Recommended next action: Proceed to S08 interface-rule tests using the S07 metric continua and exemplar rows, especially segregated/patchy, polarized-conflict, and low-activity cases, while keeping S06 context-dependent dominance as a separate explanatory layer.

## Frozen Question

Do final chimeric states fall into reproducible classes such as homogeneous, patchy, layered, polarized, oscillatory, or frozen conflict?

## Inputs

- S02 mixture-ratio runs: `{config_payload['s02RunPath']}`
- S03 spatial-arrangement runs: `{config_payload['s03RunPath']}`
- S04 goal-compatibility runs: `{config_payload['s04RunPath']}`
- S05 compatibility scores: `{config_payload['s05ScorePath']}`
- S06 dominance runs: `{config_payload['s06RunPath']}`
- S06 dominance hierarchy: `{config_payload['s06HierarchyPath']}`
- S06 context-dependence table: `{config_payload['s06ContextPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S07 harmonized S02, S03, S04, and S06 run-level outputs into one final-state table. S05 compatibility dimensions were joined as explanatory context for S02-S04 rows. S06 dominance hierarchy and context-dependent dominance were attached as separate explanatory columns and were deliberately excluded from the classifier feature set.

The classifier feature set used final target quality, reference increasing quality, goal conflict, aggregation delta, interface density, largest block fraction, label entropy, position-bias magnitude, work rate, swap rate, frozen-block rate, and sortedness delta. These are final-state or event-cap trajectory proxy measurements; no full trajectory time series was available in S02-S06.

S07 first assigned conservative rule labels, then tested whether unsupervised classifier settings supported a stable discrete taxonomy. Settings crossed spatial-only, spatial-plus-target, and spatial-plus-goal/work feature sets with KMeans and agglomerative cluster counts. Clusters were mapped back to rule labels by majority vote, then semantic label agreement was evaluated across settings. Matched-seed stability was evaluated by condition-level majority-label fractions.

Classification mode selected by validation: `{mode}`.

## Commands

- Main command: `{command}`
- Unit-test command: `{unit_test_result.get('command', 'not run')}`
- Unit-test return code: `{unit_test_result.get('returnCode', 'not run')}`

## Dependencies

No new dependencies were installed. The script used repository modules plus preinstalled `pandas`, `numpy`, `pyarrow`, `matplotlib`, and `scikit-learn` {sklearn.__version__}.

## Parameters

```json
{json.dumps(config_payload, indent=2, sort_keys=True, default=str)}
```

## Results

- Harmonized run rows: {len(mosaic)}.
- Source row counts: {mosaic['source_research_step_id'].value_counts().sort_index().to_dict()}.
- S07 labels: {mosaic['s07_label'].value_counts().sort_values(ascending=False).to_dict()}.
- Classification mode: `{mode}`.
- Stable condition fraction across seeds: {stability_context['stable_condition_fraction']:.4f}.
- Mean condition majority fraction: {stability_context['mean_majority_fraction']:.4f}.
- Classifier setting count: {stability_context['setting_count']}.
- Mean pairwise semantic agreement across classifier settings: {stability_context['mean_pairwise_semantic_agreement']:.4f}.
- Minimum pairwise semantic agreement across classifier settings: {stability_context['min_pairwise_semantic_agreement']:.4f}.
- Mean agreement between classifier settings and rule labels: {stability_context['mean_rule_agreement']:.4f}.

Class summary:

{markdown_table(class_display, ["s07_label", "class_family", "run_count", "condition_count", "mean_final_target_quality", "mean_goal_conflict_index", "mean_aggregation_delta_percent", "mean_largest_block_fraction", "mean_position_bias_abs", "mean_mosaic_continuum_score", "mean_conflict_continuum_score"], max_rows=30)}

Metric continua:

{markdown_table(continua_display, ["metric", "min", "p10", "median", "p90", "max", "mean", "std"], max_rows=20)}

Least stable condition groups:

{markdown_table(unstable, ["source_research_step_id", "analysis_condition_id", "run_count", "seed_count", "majority_label", "majority_fraction", "label_counts_json"], max_rows=15)}

Representative exemplar rows:

{markdown_table(exemplar_display, ["s07_label", "source_research_step_id", "display_names", "ratio_label", "arrangement", "goal_profile_id", "final_target_quality", "goal_conflict_index", "aggregation_delta_percent", "largest_block_fraction", "position_bias_abs"], max_rows=20)}

Classifier-setting summary:

{markdown_table(classifier_settings.sort_values("agreement_with_rule_label", ascending=False), ["setting_id", "method", "feature_set", "k", "mapped_label_count", "agreement_with_rule_label"], max_rows=20)}

Pairwise classifier-setting agreement summary:

- Pairwise rows: {len(classifier_pairwise)}.
- Mean semantic agreement: {stability_context['mean_pairwise_semantic_agreement']:.4f}.
- Minimum semantic agreement: {stability_context['min_pairwise_semantic_agreement']:.4f}.

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=30)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- S07 uses final event-cap states and trajectory proxies, not complete trajectories. Oscillation and stable turnover require future trace capture.
- The class labels are descriptive computational proxies; they should be read together with the metric continua and exemplar figure.
- S06 context-dependent dominance is retained as an explanatory layer and was not used to assign S07 labels.
- The pooled S02-S06 table combines different upstream sweeps, so source-step stratification remains important.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Run S08 interface-rule experiments using S07 exemplar rows and metric-continuum extremes as targets. Treat explicit interface rules as exploratory mechanism tests, not as evidence that prior aggregation required biological self/non-self recognition.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_mosaic_classes.parquet"
    phase_matrix_path = artifacts_dir / "results/e06_chimerism_phase_matrix.parquet"
    class_summary_path = artifacts_dir / "tables/e06_mosaic_class_summary.csv"
    continua_path = artifacts_dir / "tables/e06_s07_metric_continua.csv"
    condition_stability_path = artifacts_dir / "tables/e06_s07_condition_stability.csv"
    exemplar_path = step_dir / "e06_s07_exemplar_rows.csv"
    classifier_settings_path = step_dir / "e06_s07_classifier_settings.csv"
    classifier_pairwise_path = step_dir / "e06_s07_classifier_pairwise_agreement.csv"
    pca_summary_path = step_dir / "e06_s07_pca_summary.csv"
    validation_path = step_dir / "e06_s07_validation_checks.csv"
    figure_path = artifacts_dir / "figures/e06/mosaic_class_examples.png"
    config_path = artifacts_dir / "configs/e06_s07_mosaic_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S07Config(
        seed_stability_threshold=args.seed_stability_threshold,
        classifier_stability_threshold=args.classifier_stability_threshold,
        classifier_min_pairwise_agreement=args.classifier_min_pairwise_agreement,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "s02RunPath": str(args.s02_run_path),
        "s03RunPath": str(args.s03_run_path),
        "s04RunPath": str(args.s04_run_path),
        "s05ScorePath": str(args.s05_score_path),
        "s06RunPath": str(args.s06_run_path),
        "s06HierarchyPath": str(args.s06_hierarchy_path),
        "s06ContextPath": str(args.s06_context_path),
        "seedStabilityThreshold": config.seed_stability_threshold,
        "classifierStabilityThreshold": config.classifier_stability_threshold,
        "classifierMinPairwiseAgreement": config.classifier_min_pairwise_agreement,
        "classifierKValues": list(config.classifier_k_values),
        "classifierRandomStates": list(config.classifier_random_states),
        "classifierFeatureColumns": list(CLASSIFIER_FEATURE_COLUMNS),
        "s06DominanceContextUsedAsClassifierFeature": False,
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    input_paths = {
        "S02": args.s02_run_path,
        "S03": args.s03_run_path,
        "S04": args.s04_run_path,
        "S06": args.s06_run_path,
    }
    for path in [*input_paths.values(), args.s05_score_path, args.s06_hierarchy_path, args.s06_context_path]:
        if not path.exists():
            raise FileNotFoundError(f"required S07 input is missing: {path}")
    frames = {step: pd.read_parquet(path) for step, path in input_paths.items()}
    input_counts = {step: int(len(frame)) for step, frame in frames.items()}
    compatibility_scores = pd.read_parquet(args.s05_score_path)
    dominance_hierarchy = pd.read_csv(args.s06_hierarchy_path)
    dominance_context = pd.read_csv(args.s06_context_path)

    mosaic = harmonize_all_runs(frames)
    mosaic = attach_explanatory_contexts(mosaic, compatibility_scores, dominance_hierarchy, dominance_context)
    mosaic = add_rule_labels(mosaic)
    mosaic, pca_summary = add_continuum_coordinates(mosaic)
    mosaic, classifier_settings, classifier_pairwise, classifier_summary = run_classifier_settings(mosaic, config)
    condition_stability, seed_summary = condition_seed_stability(mosaic, config)
    mode = classification_mode(seed_summary, classifier_summary)
    mosaic["s07_classification_mode"] = mode
    class_summary = summarize_classes(mosaic)
    continua = metric_continua_summary(mosaic)
    exemplars = select_exemplars(mosaic, config)
    stability_context = {
        **seed_summary,
        **classifier_summary,
        "classification_mode": mode,
    }
    figure_written = write_mosaic_figure(figure_path, mosaic, exemplars)
    validation = validate_s07_outputs(
        mosaic,
        input_counts,
        class_summary,
        continua,
        condition_stability,
        classifier_settings,
        classifier_pairwise,
        exemplars,
        stability_context,
        config,
        figure_written=figure_written,
        unit_tests_success=bool(unit_test_result.get("success", True)),
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.to_parquet(results_path, index=False)
    phase_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.to_parquet(phase_matrix_path, index=False)
    class_summary_path.parent.mkdir(parents=True, exist_ok=True)
    class_summary.to_csv(class_summary_path, index=False)
    continua_path.parent.mkdir(parents=True, exist_ok=True)
    continua.to_csv(continua_path, index=False)
    condition_stability_path.parent.mkdir(parents=True, exist_ok=True)
    condition_stability.to_csv(condition_stability_path, index=False)
    exemplar_path.parent.mkdir(parents=True, exist_ok=True)
    exemplars.to_csv(exemplar_path, index=False)
    classifier_settings.to_csv(classifier_settings_path, index=False)
    classifier_pairwise.to_csv(classifier_pairwise_path, index=False)
    pca_summary.to_csv(pca_summary_path, index=False)
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
        artifact_entry(results_path, artifacts_dir, "S07 run-level mosaic/final-state class table."),
        artifact_entry(phase_matrix_path, artifacts_dir, "S07 combined E06 chimerism phase matrix."),
        artifact_entry(figure_path, artifacts_dir, "S07 mosaic class exemplar and continuum figure."),
        artifact_entry(class_summary_path, artifacts_dir, "S07 mosaic class summary table."),
        artifact_entry(continua_path, artifacts_dir, "S07 metric-continuum summary table."),
        artifact_entry(condition_stability_path, artifacts_dir, "S07 matched-seed condition stability table."),
        artifact_entry(exemplar_path, artifacts_dir, "S07 representative exemplar row table."),
        artifact_entry(classifier_settings_path, artifacts_dir, "S07 classifier-setting audit table."),
        artifact_entry(classifier_pairwise_path, artifacts_dir, "S07 pairwise classifier-setting agreement table."),
        artifact_entry(pca_summary_path, artifacts_dir, "S07 continuum PCA summary table."),
        artifact_entry(validation_path, artifacts_dir, "S07 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S07 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S07 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S07 full-results Markdown handoff report.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        mosaic=mosaic,
        class_summary=class_summary,
        continua=continua,
        condition_stability=condition_stability,
        classifier_settings=classifier_settings,
        classifier_pairwise=classifier_pairwise,
        exemplars=exemplars,
        validation=validation,
        config_payload=config_payload,
        stability_context=stability_context,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S07 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    manifest = {
        "schema": "eidosoma.e06.s07_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "runCount": int(len(mosaic)),
        "sourceCounts": input_counts,
        "classCount": int(mosaic["s07_label"].nunique()),
        "classificationMode": mode,
        "stableConditionFraction": float(stability_context["stable_condition_fraction"]),
        "meanPairwiseSemanticAgreement": float(stability_context["mean_pairwise_semantic_agreement"]),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S07 artifact manifest.")
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
            "s02Runs": source_entry(args.s02_run_path),
            "s03Runs": source_entry(args.s03_run_path),
            "s04Runs": source_entry(args.s04_run_path),
            "s05Scores": source_entry(args.s05_score_path),
            "s06Runs": source_entry(args.s06_run_path),
            "s06Hierarchy": source_entry(args.s06_hierarchy_path),
            "s06Context": source_entry(args.s06_context_path),
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        phase_matrix_path,
        figure_path,
        class_summary_path,
        continua_path,
        condition_stability_path,
        exemplar_path,
        classifier_settings_path,
        classifier_pairwise_path,
        pca_summary_path,
        validation_path,
        config_path,
        manifest_path,
        report_path,
        run_manifest_path,
    ]
    write_text(checksums_path, "\n".join(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_paths) + "\n")

    status = {
        "success": success,
        "runCount": int(len(mosaic)),
        "sourceCounts": input_counts,
        "classCount": int(mosaic["s07_label"].nunique()),
        "classificationMode": mode,
        "stableConditionFraction": float(stability_context["stable_condition_fraction"]),
        "meanPairwiseSemanticAgreement": float(stability_context["mean_pairwise_semantic_agreement"]),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
