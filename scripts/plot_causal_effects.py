#!/usr/bin/env python3
"""Render compact S12 marginal-effect and heterogeneity summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT = Path("/artifacts/research_steps/S12")
ESTIMAND_LABELS = {
    "E02-S01-E04": "E04 continuation",
    "E02-S01-E06": "E06 stuck vs passive",
    "E02-S01-E03": "E03 scheduler",
    "E02-S01-E08": "E08 weak coordinator",
}
ENDPOINT_LABELS = {
    "normalizedResidualError": "Normalized residual error",
    "successByBudget": "Success risk",
    "completionCumulativeIncidenceAtBudget": "Completion CIF at budget",
    "restrictedMeanCompletionFreeBudgetFraction": "Restricted mean completion-free fraction",
    "logS01UnitWeightFullCost": "Log S01 unit-weight cost",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    marginal = pd.read_parquet(args.output / "marginal_effects.parquet")
    order = list(ESTIMAND_LABELS)
    endpoints = list(ENDPOINT_LABELS)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    axes = axes.ravel()
    colors = ["#31688e", "#35b779", "#fde725", "#443983"]
    for axis, endpoint in zip(axes, endpoints, strict=False):
        part = marginal[marginal.endpoint == endpoint].set_index("estimandId").loc[order]
        y = list(range(len(order)))
        axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        for index, (estimand, row) in enumerate(part.iterrows()):
            axis.errorbar(
                row.estimate, index,
                xerr=[[row.estimate - row.bootstrapLow95], [row.bootstrapHigh95 - row.estimate]],
                fmt="o", color=colors[index], capsize=3, linewidth=1.5,
            )
        axis.set_yticks(y, [ESTIMAND_LABELS[item] for item in order])
        axis.invert_yaxis()
        axis.set_title(ENDPOINT_LABELS[endpoint], fontsize=10)
        axis.set_xlabel("Active minus reference (95% paired bootstrap interval)", fontsize=8)
        axis.grid(axis="x", alpha=0.2)
    axes[-1].axis("off")
    figure.suptitle(
        "S12 standardized causal total effects within S11 protected support",
        fontsize=14,
    )
    figure.savefig(args.output / "marginal_effects.png", dpi=180)
    figure.savefig(args.output / "marginal_effects.svg")
    plt.close(figure)

    rankings = pd.read_parquet(args.output / "heterogeneity_rankings.parquet")
    selected = rankings[
        (rankings.associationalRangeRank == 1)
        & rankings.endpoint.isin([
            "normalizedResidualError", "successByBudget", "logS01UnitWeightFullCost"
        ])
    ].copy()
    selected["label"] = selected.estimandId.map(ESTIMAND_LABELS)
    ranked_endpoints = [
        "normalizedResidualError", "successByBudget", "logS01UnitWeightFullCost"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    for axis, endpoint in zip(axes, ranked_endpoints, strict=True):
        part = selected[selected.endpoint == endpoint].sort_values("effectRange")
        axis.barh(part.label, part.effectRange, color="#31688e")
        for y, row in enumerate(part.itertuples(index=False)):
            axis.text(row.effectRange, y, f"  {row.modifier}", va="center", fontsize=8)
        axis.set_xlabel("Largest supported one-way effect range", fontsize=8)
        axis.set_title(ENDPOINT_LABELS[endpoint], fontsize=10)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Prespecified associational effect-modification ranking\n"
        "(ranges are comparable only within endpoint; labels show the leading modifier)",
        fontsize=13,
    )
    figure.savefig(args.output / "heterogeneity_leading_ranges.png", dpi=180)
    figure.savefig(args.output / "heterogeneity_leading_ranges.svg")
    plt.close(figure)
    print("wrote 4 figure files")


if __name__ == "__main__":
    main()
