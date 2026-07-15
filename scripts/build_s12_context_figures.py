#!/usr/bin/env python3
"""Build compact S12 context, threshold, segmentation, and calibration figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OUTPUT = Path("/artifacts/research_steps/S12")
METRICS = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
COLORS = {
    "adjacent_descents": "#3366cc",
    "inversion_count": "#dc3912",
    "spearman_footrule": "#109618",
    "maximum_rank_error": "#990099",
}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def context_figure(effects: pd.DataFrame) -> None:
    view = effects[
        effects.endpoint.eq("detour_occurrence")
        & effects.factor.isin(["n", "scheduler_profile", "initial_disorder_profile", "intervention_type"])
    ]
    factors = ("n", "scheduler_profile", "initial_disorder_profile", "intervention_type")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, factor in zip(axes.ravel(), factors, strict=True):
        subset = view[view.factor.eq(factor)]
        levels = subset.level.drop_duplicates().tolist()
        x = np.arange(len(levels), dtype=float)
        for metric_index, metric in enumerate(METRICS):
            rows = subset[subset.metric.eq(metric)].set_index("level").reindex(levels)
            offset = (metric_index - 1.5) * 0.16
            axis.errorbar(
                x + offset,
                rows.estimate,
                yerr=[rows.estimate - rows.ci_low, rows.ci_high - rows.estimate],
                marker="o",
                capsize=2,
                linewidth=1.2,
                color=COLORS[metric],
                label=metric.replace("_", " "),
            )
        axis.set_xticks(x, levels, rotation=25, ha="right")
        axis.set_ylabel("Detour occurrence")
        axis.set_title(factor.replace("_", " ").title())
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2, axis="y")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("S12 empirical larger-n context effects (cluster bootstrap 95% intervals)")
    save(fig, "context_marginal_effects")


def threshold_figure(thresholds: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for metric in METRICS:
        rows = thresholds[thresholds.metric.eq(metric)].sort_values("threshold_raw_units")
        axis.plot(
            rows.threshold_raw_units,
            rows.exposure_rate,
            marker="o",
            color=COLORS[metric],
            label=metric.replace("_", " "),
        )
    axis.set_xlabel("Strict worsening threshold (raw metric units)")
    axis.set_ylabel("Runs with ≥1 accepted proposal above threshold")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    axis.set_title("Metric-specific S10 exposure sensitivity")
    save(fig, "threshold_plots")


def segmentation_figure(segmentation: pd.DataFrame) -> None:
    view = (
        segmentation.groupby(["metric", "segmentation_definition"], observed=True)
        .apply(lambda group: np.average(group.rate, weights=group.runs), include_groups=False)
        .rename("rate")
        .reset_index()
    )
    definitions = view.segmentation_definition.drop_duplicates().tolist()
    x = np.arange(len(definitions))
    fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    width = 0.18
    for index, metric in enumerate(METRICS):
        rows = view[view.metric.eq(metric)].set_index("segmentation_definition").reindex(definitions)
        axis.bar(x + (index - 1.5) * width, rows.rate, width, color=COLORS[metric], label=metric.replace("_", " "))
    axis.set_xticks(x, [item.replace("_", " ") for item in definitions], rotation=20, ha="right")
    axis.set_ylabel("Positive-run fraction")
    axis.set_ylim(0, 1)
    axis.set_title("Episode-definition sensitivity")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2, axis="y")
    save(fig, "episode_segmentation_sensitivity")


def calibration_figure(folds: pd.DataFrame | None) -> None:
    if folds is None:
        fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.58,
            "Held-out calibration not estimable",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )
        axis.text(
            0.5,
            0.38,
            "The frozen multi-metric family hit its predeclared gate:\n"
            "inversion, footrule, and maximum-rank outcomes each had zero detour events.\n"
            "No reduced one-metric prediction model was substituted.",
            ha="center",
            va="center",
            fontsize=11,
        )
        save(fig, "held_out_calibration")
        return
    binary = folds[folds.brier.notna()].copy()
    model_order = binary.model_id.drop_duplicates().tolist()
    summary = binary.groupby("model_id", observed=True).agg(
        brier=("brier", "mean"), ece=("ece_10", "mean"), slope=("calibration_slope", "mean")
    ).reindex(model_order)
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(x - 0.18, summary.brier, 0.36, label="Brier", color="#3366cc")
    axes[0].bar(x + 0.18, summary.ece, 0.36, label="ECE", color="#ff9900")
    axes[0].set_xticks(x, [value.replace("occurrence_", "").replace("_", " ") for value in summary.index], rotation=25, ha="right")
    axes[0].set_title("Held-out probabilistic error")
    axes[0].legend()
    axes[0].grid(alpha=0.2, axis="y")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].plot(x, summary.slope, "o", color="#109618")
    axes[1].set_xticks(x, [value.replace("occurrence_", "").replace("_", " ") for value in summary.index], rotation=25, ha="right")
    axes[1].set_ylabel("Calibration slope")
    axes[1].set_title("Held-out calibration")
    axes[1].grid(alpha=0.2, axis="y")
    save(fig, "held_out_calibration")


def main() -> None:
    effects = pq.read_table(OUTPUT / "marginal_effects.parquet").to_pandas()
    thresholds = pq.read_table(OUTPUT / "threshold_sensitivity.parquet").to_pandas()
    segmentation = pq.read_table(OUTPUT / "episode_segmentation_sensitivity.parquet").to_pandas()
    folds_path = OUTPUT / "context_models/held_out_prediction_by_fold.parquet"
    folds = pq.read_table(folds_path).to_pandas() if folds_path.exists() else None
    context_figure(effects)
    threshold_figure(thresholds)
    segmentation_figure(segmentation)
    calibration_figure(folds)


if __name__ == "__main__":
    main()
