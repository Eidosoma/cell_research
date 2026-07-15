#!/usr/bin/env python3
"""Build compact S11 null-distribution summaries and the canonical figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detours.barrier_interventions import METRICS
from src.detours.behavioral_nulls import NULL_NAMES
from scripts.build_s11_behavioral_nulls import OUTPUT, write_parquet


S09 = Path("/artifacts/research_steps/S09")


SHORT = {
    "random_legal": "Random legal",
    "rate_matched_random": "Rate matched",
    "open_loop_opportunity": "Open loop",
    "greedy_adjacent_descents": "Greedy adjacent",
    "greedy_inversion_count": "Greedy inversion",
    "greedy_spearman_footrule": "Greedy footrule",
    "greedy_maximum_rank_error": "Greedy max-rank",
}


def main() -> None:
    nulls = pd.read_parquet(OUTPUT / "null_results.parquet")
    comparisons = pd.read_parquet(OUTPUT / "observed_null_comparisons.parquet")
    matching = pd.read_parquet(OUTPUT / "action_rate_matching.parquet")
    rows = []
    for null_name, group in nulls.groupby("null_family", sort=False):
        rows.append(
            {
                "null_family": null_name,
                "run_count": len(group),
                "completion_count": int(group.completed.sum()),
                "completion_rate": float(group.completed.mean()),
                "quiescent_stop_count": int(group.stop_reason.eq("quiescent").sum()),
                "event_budget_count": int(group.stop_reason.eq("event_budget").sum()),
                "mean_event_count": float(group.event_count.mean()),
                "mean_accepted_swaps": float(group.cost_acceptedSwaps.mean()),
                "mean_s06_full_ledger_cost": float(group.s06_full_ledger_cost.mean()),
            }
        )
    summary = pd.DataFrame(rows)
    write_parquet(
        OUTPUT / "null_summary_by_family.parquet",
        summary,
        "e03.s11.null_summary_by_family.v1",
    )
    barrier = pd.read_parquet(
        OUTPUT / "null_barrier_effects.parquet",
        filters=[("metric", "=", "adjacent_descents")],
    )
    barrier_rows = []
    for (null_name, intervention), group in barrier.groupby(
        ["null_family", "intervention_type"], sort=True
    ):
        barrier_rows.append(
            {
                "policy": null_name,
                "intervention_type": intervention,
                "pair_count": len(group),
                "mean_completion_difference_arm_minus_baseline": float(
                    group.delta_completed_arm_minus_baseline.mean()
                ),
                "both_successful_count": int(
                    group.successful_efficiency_comparable.sum()
                ),
                "mean_activation_difference_when_both_successful": float(
                    group.delta_activations_when_both_successful.mean()
                ),
            }
        )
    observed = pd.read_parquet(S09 / "barrier_interventions.parquet")
    observed = observed[
        observed.intervention_type.isin(["baseline", "remove", "move", "add"])
    ]
    baseline = observed[observed.intervention_type == "baseline"][
        ["pair_block_id", "completed", "event_count"]
    ].rename(
        columns={"completed": "baseline_completed", "event_count": "baseline_event_count"}
    )
    arms = observed[observed.intervention_type != "baseline"].merge(
        baseline, on="pair_block_id", validate="many_to_one"
    )
    arms["completion_difference"] = (
        arms.completed.astype(np.int8) - arms.baseline_completed.astype(np.int8)
    )
    arms["both_successful"] = arms.completed & arms.baseline_completed
    arms["activation_difference"] = (
        arms.event_count - arms.baseline_event_count
    ).where(arms.both_successful)
    for intervention, group in arms.groupby("intervention_type", sort=True):
        barrier_rows.append(
            {
                "policy": "observed_s09",
                "intervention_type": intervention,
                "pair_count": len(group),
                "mean_completion_difference_arm_minus_baseline": float(
                    group.completion_difference.mean()
                ),
                "both_successful_count": int(group.both_successful.sum()),
                "mean_activation_difference_when_both_successful": float(
                    group.activation_difference.mean()
                ),
            }
        )
    write_parquet(
        OUTPUT / "barrier_contingency_summary.parquet",
        barrier_rows,
        "e03.s11.barrier_contingency_summary.v1",
    )

    order = list(NULL_NAMES)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    plot_summary = summary.set_index("null_family").loc[order]
    ax.barh(range(len(order)), plot_summary.completion_rate, color=colors)
    ax.set_yticks(range(len(order)), [SHORT[name] for name in order])
    ax.invert_yaxis()
    ax.set_xlabel("Completion proportion by 2,048 opportunities")
    ax.set_title("A. Completion under each structural null")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[0, 1]
    status = (
        nulls.assign(
            terminal_status=np.select(
                [nulls.completed, nulls.stop_reason.eq("quiescent")],
                ["complete", "quiescent"],
                default="active at budget",
            )
        )
        .groupby(["null_family", "terminal_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    status = status.div(status.sum(axis=1), axis=0)
    left = np.zeros(len(order))
    for label, color in zip(
        ("complete", "quiescent", "active at budget"),
        ("#2A9D8F", "#B8B8B8", "#E76F51"),
    ):
        values = status[label].to_numpy()
        ax.barh(range(len(order)), values, left=left, label=label, color=color)
        left += values
    ax.set_yticks(range(len(order)), [SHORT[name] for name in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Run proportion")
    ax.set_title("B. Terminal-state accounting")
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
    )

    ax = axes[1, 0]
    heat = (
        comparisons.groupby(["null_family", "metric"])
        .delta_final_error_null_minus_observed.mean()
        .unstack()
        .reindex(index=order, columns=list(METRICS))
    )
    scale = float(np.nanmax(np.abs(heat.to_numpy())))
    image = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    ax.set_yticks(range(len(order)), [SHORT[name] for name in order])
    ax.set_xticks(
        range(len(METRICS)),
        ["Adjacent", "Inversion", "Footrule", "Max-rank"],
        rotation=25,
        ha="right",
    )
    for row in range(len(order)):
        for column in range(len(METRICS)):
            ax.text(column, row, f"{heat.iloc[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85, label="Mean null − observed final error")
    ax.set_title("C. Residual error relative to matched S09 behavior")

    ax = axes[1, 1]
    deficit = (
        matching.infeasible_swap_request_count
        + matching.infeasible_memory_request_count
        + matching.infeasible_unchanged_request_count
        > 0
    )
    sample = matching.sample(n=min(12_000, len(matching)), random_state=1103)
    sample_deficit = deficit.loc[sample.index]
    ax.scatter(
        sample.loc[~sample_deficit, "rate_matched_target_swap_rate"],
        sample.loc[~sample_deficit, "achieved_swap_rate"],
        s=7,
        alpha=0.25,
        color="#4C78A8",
        label="all requested categories feasible",
    )
    ax.scatter(
        sample.loc[sample_deficit, "rate_matched_target_swap_rate"],
        sample.loc[sample_deficit, "achieved_swap_rate"],
        s=7,
        alpha=0.25,
        color="#E45756",
        label="at least one category unavailable",
    )
    limit = max(
        float(sample.rate_matched_target_swap_rate.max()),
        float(sample.achieved_swap_rate.max()),
        0.01,
    )
    ax.plot([0, limit], [0, limit], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(-0.01, limit * 1.02)
    ax.set_ylim(-0.01, limit * 1.02)
    ax.set_xlabel("Target accepted-swap/opportunity rate")
    ax.set_ylabel("Achieved accepted-swap/opportunity rate")
    ax.set_title("D. Rate matching and feasibility boundary")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    fig.suptitle(
        "E03 S11 behavioral nulls: outcomes, residuals, and rate calibration",
        fontsize=15,
    )
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"null_distributions.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
