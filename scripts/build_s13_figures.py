#!/usr/bin/env python3
"""Build compact S13 effect, metric-event, complexity, and trace figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT = Path("/artifacts/research_steps/S13")
ARM_ORDER = [
    "native", "failure_bit", "counter_2bit", "recent_direction", "radius2",
    "full", "full_minus_failure", "full_minus_counter", "full_minus_recent",
    "full_minus_radius2",
]
METRIC_LABELS = {
    "adjacent_descents": "Adjacent descents",
    "inversion_count": "Inversions",
    "spearman_footrule": "Footrule",
    "maximum_rank_error": "Max rank error",
}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def write_panel(frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": b"e03.s13.representative_trace_panel.v1", b"researchStepId": b"S13"}
    )
    pq.write_table(table, OUTPUT / "representative_trace_panel.parquet", compression="zstd", compression_level=9)


def main() -> None:
    effects = pq.read_table(OUTPUT / "ablation_effect_summary.parquet").to_pandas()
    capability = pq.read_table(OUTPUT / "capability_summary.parquet").to_pandas()
    pareto = pq.read_table(OUTPUT / "complexity_effect_pareto.parquet").to_pandas()
    traces = pq.read_table(OUTPUT / "retained_metric_traces.parquet").to_pandas()
    metrics = pq.read_table(OUTPUT / "memory_information_results.parquet").to_pandas()

    completion = effects[
        effects.metric.eq("adjacent_descents")
        & effects.endpoint.eq("completion_risk_difference")
    ].copy()
    contrast_order = [
        "single_failure_bit", "single_counter_2bit", "single_recent_direction", "single_radius2",
        "full_vs_native", "ablate_failure", "ablate_counter", "ablate_recent", "ablate_radius2",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for axis, tier, title in zip(
        axes,
        ["empirical_large_n", "exact_small_n_anchor_empirical_path"],
        ["Empirical n=12–24", "Exact-labelled n=4–5 starts"],
        strict=True,
    ):
        frame = completion[completion.evidence_tier.eq(tier)].set_index("contrast_id").loc[contrast_order]
        y = np.arange(len(frame))
        mean = frame.mean_difference_first_minus_second.to_numpy()
        axis.errorbar(mean, y, xerr=[mean - frame.ci_low, frame.ci_high - mean], fmt="o", color="#315b7d", capsize=3)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, contrast_order if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Completion risk difference (first − second)")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Frozen paired capability contrasts (2,000 block bootstraps)")
    fig.tight_layout()
    save(fig, "memory_information_effects")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, tier, title in zip(
        axes,
        ["empirical_large_n", "exact_small_n_anchor_empirical_path"],
        ["Empirical n=12–24", "Exact-labelled n=4–5 starts"],
        strict=True,
    ):
        frame = capability[capability.evidence_tier.eq(tier)].pivot(
            index="capability_arm", columns="metric", values="detour_rate"
        ).loc[ARM_ORDER, list(METRIC_LABELS)]
        image = axis.imshow(frame.to_numpy(), vmin=0, vmax=1, cmap="magma", aspect="auto")
        axis.set_xticks(np.arange(4), [METRIC_LABELS[x] for x in frame.columns], rotation=35, ha="right")
        axis.set_yticks(np.arange(len(frame)), frame.index if axis is axes[0] else [])
        axis.set_title(title)
    fig.colorbar(image, ax=axes, label="Run-level detour occurrence")
    fig.suptitle("Metric-specific events remain separate")
    save(fig, "metric_event_heatmap")

    frame = pareto[pareto.evidence_tier.eq("empirical_large_n")].copy()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(frame.bits_plus_radius, frame.mean_difference_first_minus_second, s=80, c=frame.absolute_completion_effect, cmap="viridis")
    for row in frame.itertuples(index=False):
        axis.annotate(row.arm, (row.bits_plus_radius, row.mean_difference_first_minus_second), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Declared stored bits per actor + sensing radius")
    axis.set_ylabel("Completion difference versus contrast reference")
    axis.set_title("Complexity is explicit; it is not a monotone benefit scale")
    axis.grid(alpha=0.25)
    save(fig, "complexity_pareto")

    retained_ids = set(traces.run_id.unique())
    candidates = metrics[
        metrics.run_id.isin(retained_ids)
        & metrics.metric.isin(["inversion_count", "adjacent_descents"])
    ].copy()
    specifications = [
        ("empirical_large_n", "radius2", "inversion_count"),
        ("empirical_large_n", "full", "inversion_count"),
        ("empirical_large_n", "full_minus_recent", "inversion_count"),
        ("empirical_large_n", "native", "adjacent_descents"),
        ("empirical_large_n", "recent_direction", "adjacent_descents"),
        ("exact_small_n_anchor_empirical_path", "native", "inversion_count"),
    ]
    selected = []
    for tier, arm, metric in specifications:
        frame = candidates[
            candidates.evidence_tier.eq(tier) & candidates.capability_arm.eq(arm)
            & candidates.metric.eq(metric)
        ].sort_values(["max_episode_depth", "running_min_episode_count", "run_id"], ascending=[False, False, True])
        if len(frame):
            selected.append((frame.iloc[0].run_id, metric, tier, arm))
    panels = []
    for run_id, metric, tier, arm in selected:
        frame = traces[traces.run_id.eq(run_id)].copy()
        frame["panel_metric"] = metric
        frame["panel_distance"] = frame[f"distance_{metric}"]
        frame["panel_label"] = f"{tier}: {arm}: {METRIC_LABELS[metric]}"
        panels.append(frame)
    panel = pd.concat(panels, ignore_index=True)
    write_panel(panel)
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.2 * len(selected)), squeeze=False)
    for axis, (_, group) in zip(axes[:, 0], panel.groupby("panel_label", sort=False), strict=True):
        axis.step(group.event_count, group.panel_distance, where="post", color="#8c2d3c")
        axis.set_title(group.panel_label.iloc[0], loc="left", fontsize=9)
        axis.set_ylabel("Distance")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Charged opportunities")
    fig.suptitle("Representative retained metric-checkpoint traces")
    fig.tight_layout()
    save(fig, "representative_metric_traces")


if __name__ == "__main__":
    main()
