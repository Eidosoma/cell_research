#!/usr/bin/env python3
"""Render the report-review Pareto figure from the frozen S14 summary.

This presentation-only renderer leaves the frozen S14 analysis tables unchanged.
It labels only configurations meeting the prespecified robust-inclusion threshold
and reconstructs labels from the source configuration fields so terminology is
not inherited from abbreviated plotting labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path("/artifacts/research_steps/S14/pareto_summary.parquet")
DEFAULT_DESTINATION = Path(
    "/artifacts/generatedPapersAndSummaries/20260715T111617Z/figures/"
    "falsifymap_pareto_map.png"
)


def source_setting_label(record: Mapping[str, Any]) -> str:
    """Return a readable label that retains the frozen source terminology."""
    architecture = {
        "distributed_local": "distributed",
        "distributed_weak_coordinator": "weak",
        "central_local_proposal_k1": "central-local-k1",
    }.get(str(record["architecture"]), str(record["architecture"]))
    scheduler = {
        "uniform_random_activation": "uniform",
        "random_permutation_sweep": "permutation",
    }.get(str(record["scheduler"]), str(record["scheduler"]))
    continuation = {
        "skip_and_continue": "skip-and-continue",
        "stop_on_first_blocking_failure": "stop-on-first-blocking-failure",
    }.get(str(record["continuation"]), str(record["continuation"]))
    coordinator = str(record["coordinatorProfile"])
    parts = [architecture]
    if coordinator != "none":
        parts.append("coordinated")
    parts.extend([scheduler, str(record["mobility"]), continuation])
    return " / ".join(parts)


def render_pareto(source: Path, destination: Path) -> dict[str, Any]:
    summary = pd.read_parquet(source)
    required = {
        "treatmentSignature",
        "architecture",
        "coordinatorProfile",
        "scheduler",
        "mobility",
        "continuation",
        "costProfile",
        "meanNormalizedResidual",
        "failureRate",
        "meanLogCost",
        "robustPareto",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"missing Pareto columns: {missing}")
    primary = summary[summary.costProfile == "s01Cost"].copy()
    if primary.empty or primary.treatmentSignature.duplicated().any():
        raise ValueError("expected one primary-cost row per treatment signature")
    numeric = primary[["meanNormalizedResidual", "failureRate", "meanLogCost"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Pareto display values must be finite")

    primary["reportLabel"] = [
        source_setting_label(record) for record in primary.to_dict("records")
    ]
    robust = primary[primary.robustPareto.astype(bool)].sort_values(
        ["meanNormalizedResidual", "failureRate", "meanLogCost"]
    )
    other = primary[~primary.robustPareto.astype(bool)]
    if robust.empty:
        raise ValueError("the frozen summary contains no robustly included setting")

    minimum = float(primary.meanLogCost.min())
    span = max(float(primary.meanLogCost.max() - minimum), 1e-12)

    def marker_sizes(frame: pd.DataFrame) -> np.ndarray:
        return 55 + 95 * (frame.meanLogCost.to_numpy(dtype=float) - minimum) / span

    figure, axis = plt.subplots(figsize=(11.8, 6.2))
    if not other.empty:
        axis.scatter(
            other.meanNormalizedResidual,
            other.failureRate,
            s=marker_sizes(other),
            color="#7599b8",
            alpha=0.58,
            edgecolor="white",
            linewidth=0.8,
            label="Other S11-supported configurations",
        )

    palette = ["#2d6a9f", "#d07c2c", "#b23a48", "#4b8b3b"]
    markers = ["o", "s", "D", "^"]
    for index, row in enumerate(robust.itertuples(index=False)):
        size = 55 + 95 * (float(row.meanLogCost) - minimum) / span
        axis.scatter(
            [row.meanNormalizedResidual],
            [row.failureRate],
            s=[size],
            color=palette[index % len(palette)],
            marker=markers[index % len(markers)],
            edgecolor="black",
            linewidth=0.9,
            alpha=0.92,
            label=row.reportLabel,
        )

    axis.set_xlabel("Mean normalized residual (lower is better)")
    axis.set_ylabel("Failure rate (lower is better)")
    axis.set_title(
        "S11-supported empirical Pareto map\n"
        "Labels identify configurations with Pareto-inclusion probability ≥ 0.90"
    )
    axis.grid(alpha=0.22)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
        fontsize=8.5,
        title="Frozen candidate set",
        title_fontsize=9,
    )
    figure.text(
        0.125,
        0.015,
        "Point area increases with mean log S01 unit-weight cost; cost is a model-defined ledger projection.",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.04, 0.78, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return {
        "source": str(source),
        "destination": str(destination),
        "primarySettingCount": int(len(primary)),
        "robustlyIncludedCount": int(len(robust)),
        "labeledSettings": robust.reportLabel.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    print(json.dumps(render_pareto(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
