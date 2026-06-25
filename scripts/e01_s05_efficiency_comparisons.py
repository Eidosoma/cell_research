#!/usr/bin/env python3
"""Build E01 S05 efficiency tables and Figure 4-style plots.

S05 consumes the completed S04 no-Frozen replicate summary. It normalizes the
S04 comparison fields into explicit swap-only, comparison-only, and
swap-plus-comparison metrics, then stops before S06 statistical tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E01"
STEP_ID = "S05"
STEP_NUMBER = 5
STATUS = "completed"
OUTCOME_CLASSIFICATION = "constraining"
ALGORITHMS = ["bubble", "insertion", "selection"]
IMPLEMENTATIONS = ["traditional", "cell_view"]
METRIC_ORDER = ["swap_only_steps", "swap_plus_comparison_steps"]
S04_REPLICATE_SUMMARY_DEFAULT = Path("/artifacts/results/e01_s04_replicate_summary.parquet")
S04_STATUS_DEFAULT = Path("/artifacts/research_steps/S04/status.json")
S03_CONFIG_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
S03_SEED_TABLE_DEFAULT = Path("/artifacts/research_steps/S03/seed_table.csv")

PAPER_CLAIMS: dict[tuple[str, str], dict[str, Any]] = {
    ("swap_only_steps", "bubble"): {
        "paperRelation": "similar",
        "paperText": "Paper reports no significant swap-only difference for Bubble sort (z=0.73, p=0.47).",
        "paperReportedFactor": None,
    },
    ("swap_only_steps", "insertion"): {
        "paperRelation": "similar",
        "paperText": "Paper reports no significant swap-only difference for Insertion sort (z=1.26, p=0.24).",
        "paperReportedFactor": None,
    },
    ("swap_only_steps", "selection"): {
        "paperRelation": "cell_view_greater",
        "paperText": "Paper reports cell-view Selection needs about 11 times more swaps.",
        "paperReportedFactor": 11.0,
    },
    ("swap_plus_comparison_steps", "bubble"): {
        "paperRelation": "cell_view_fewer",
        "paperText": "Paper reports cell-view Bubble has fewer comparison-inclusive steps by 1.5 times.",
        "paperReportedFactor": 1.5,
    },
    ("swap_plus_comparison_steps", "insertion"): {
        "paperRelation": "cell_view_fewer",
        "paperText": "Paper reports cell-view Insertion has fewer comparison-inclusive steps by 2.03 times.",
        "paperReportedFactor": 2.03,
    },
    ("swap_plus_comparison_steps", "selection"): {
        "paperRelation": "cell_view_greater",
        "paperText": "Paper reports cell-view Selection has greater comparison-inclusive steps by 1.17 times.",
        "paperReportedFactor": 1.17,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"] if commit["ok"] else "unknown",
        "branch": branch["stdout"] if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"],
        "remote": remote["stdout"],
    }


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_builtin(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append(
                {
                    "path": str(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return sorted(artifacts, key=lambda item: item["path"])


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def required_columns() -> set[str]:
    return {
        "research_step_id",
        "condition_id",
        "implementation",
        "algorithm",
        "replicate_index",
        "replicate_number",
        "input_profile",
        "n",
        "input_permutation_seed",
        "scheduler_seed",
        "tie_breaker_seed",
        "initial_state_hash",
        "final_sortedness_percent",
        "final_monotonicity_error",
        "completed",
        "stop_reason",
        "swap_count",
        "comparison_count",
        "archived_compare_and_swap_count",
    }


def normalize_replicates(s04_df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(required_columns() - set(s04_df.columns))
    if missing:
        raise RuntimeError(f"S04 replicate summary is missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for row in s04_df.to_dict(orient="records"):
        implementation = str(row["implementation"])
        if implementation == "traditional":
            comparison_only_count = int(row["comparison_count"])
            comparison_source_field = "comparison_count"
            comparison_count_convention = "traditional_wrapper_value_comparisons"
        elif implementation == "cell_view":
            comparison_only_count = int(row["archived_compare_and_swap_count"])
            comparison_source_field = "archived_compare_and_swap_count"
            comparison_count_convention = "archived_status_probe_compare_and_swap_count"
        else:
            raise RuntimeError(f"unexpected implementation: {implementation}")

        swap_only = int(row["swap_count"])
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": str(row["research_step_id"]),
                "source_condition_id": str(row["condition_id"]),
                "implementation": implementation,
                "algorithm": str(row["algorithm"]),
                "replicate_index": int(row["replicate_index"]),
                "replicate_number": int(row["replicate_number"]),
                "input_profile": str(row["input_profile"]),
                "n": int(row["n"]),
                "input_permutation_seed": int(row["input_permutation_seed"]),
                "scheduler_seed": int(row["scheduler_seed"]),
                "tie_breaker_seed": int(row["tie_breaker_seed"]),
                "initial_state_hash": str(row["initial_state_hash"]),
                "final_sortedness_percent": float(row["final_sortedness_percent"]),
                "final_monotonicity_error": int(row["final_monotonicity_error"]),
                "completed": bool(row["completed"]),
                "stop_reason": str(row["stop_reason"]),
                "swap_only_steps": swap_only,
                "comparison_only_count": comparison_only_count,
                "swap_plus_comparison_steps": int(swap_only + comparison_only_count),
                "raw_s04_swap_count": swap_only,
                "raw_s04_comparison_count": int(row["comparison_count"]),
                "raw_s04_archived_compare_and_swap_count": int(row["archived_compare_and_swap_count"]),
                "comparison_source_field": comparison_source_field,
                "comparison_count_convention": comparison_count_convention,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["algorithm", "replicate_index", "implementation"]).reset_index(drop=True)


def summarize_conditions(replicate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        for (algorithm, implementation), group in replicate_df.groupby(["algorithm", "implementation"], sort=True):
            values = group[metric_id].astype(float)
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "result_family": "efficiency_condition_summary",
                    "algorithm": algorithm,
                    "implementation": implementation,
                    "metric_id": metric_id,
                    "metric_label": (
                        "Swap-only steps" if metric_id == "swap_only_steps" else "Swap plus comparison steps"
                    ),
                    "replicate_count": int(len(group)),
                    "mean_steps": float(values.mean()),
                    "sd_steps": float(values.std(ddof=1)),
                    "sem_steps": float(values.std(ddof=1) / np.sqrt(len(group))),
                    "median_steps": float(values.median()),
                    "min_steps": float(values.min()),
                    "max_steps": float(values.max()),
                    "q025_steps": float(values.quantile(0.025)),
                    "q975_steps": float(values.quantile(0.975)),
                    "comparison_count_convention": "; ".join(sorted(set(group["comparison_count_convention"]))),
                    "comparison_source_field": "; ".join(sorted(set(group["comparison_source_field"]))),
                    "source_summary_path": str(S04_REPLICATE_SUMMARY_DEFAULT),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["metric_id", "algorithm", "implementation"]).reset_index(drop=True)


def observed_direction(cell_mean: float, traditional_mean: float) -> str:
    if cell_mean < traditional_mean:
        return "cell_view_fewer"
    if cell_mean > traditional_mean:
        return "cell_view_greater"
    return "equal"


def practical_direction(cell_mean: float, traditional_mean: float, tolerance: float = 0.05) -> str:
    ratio = cell_mean / traditional_mean if traditional_mean else np.inf
    if ratio < 1.0 - tolerance:
        return "cell_view_fewer"
    if ratio > 1.0 + tolerance:
        return "cell_view_greater"
    return "similar"


def factor_for_claim(claim: dict[str, Any], cell_mean: float, traditional_mean: float) -> float | None:
    relation = claim["paperRelation"]
    if relation == "cell_view_fewer":
        return traditional_mean / cell_mean if cell_mean else np.inf
    if relation == "cell_view_greater":
        return cell_mean / traditional_mean if traditional_mean else np.inf
    return cell_mean / traditional_mean if traditional_mean else np.inf


def direction_matches_claim(claim: dict[str, Any], direction: str, practical: str) -> bool:
    relation = claim["paperRelation"]
    if relation == "similar":
        return practical == "similar"
    return direction == relation


def summarize_pairwise(replicate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        for algorithm in ALGORITHMS:
            subset = replicate_df[replicate_df["algorithm"] == algorithm]
            traditional = subset[subset["implementation"] == "traditional"][
                ["replicate_index", "input_permutation_seed", "initial_state_hash", metric_id]
            ].rename(columns={metric_id: "traditional_steps"})
            cell_view = subset[subset["implementation"] == "cell_view"][
                ["replicate_index", "input_permutation_seed", "initial_state_hash", metric_id]
            ].rename(columns={metric_id: "cell_view_steps"})
            paired = traditional.merge(
                cell_view,
                on=["replicate_index", "input_permutation_seed", "initial_state_hash"],
                how="inner",
                validate="one_to_one",
            )
            if len(paired) != 100:
                raise RuntimeError(f"expected 100 matched pairs for {algorithm} {metric_id}, found {len(paired)}")
            diff = paired["cell_view_steps"].astype(float) - paired["traditional_steps"].astype(float)
            tr = paired["traditional_steps"].astype(float)
            cv = paired["cell_view_steps"].astype(float)
            claim = PAPER_CLAIMS[(metric_id, algorithm)]
            sign_direction = observed_direction(float(cv.mean()), float(tr.mean()))
            practical = practical_direction(float(cv.mean()), float(tr.mean()))
            factor = factor_for_claim(claim, float(cv.mean()), float(tr.mean()))
            paper_factor = claim["paperReportedFactor"]
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "result_family": "efficiency_paired_comparison",
                    "algorithm": algorithm,
                    "metric_id": metric_id,
                    "metric_label": (
                        "Swap-only steps" if metric_id == "swap_only_steps" else "Swap plus comparison steps"
                    ),
                    "pair_count": int(len(paired)),
                    "traditional_mean_steps": float(tr.mean()),
                    "traditional_sd_steps": float(tr.std(ddof=1)),
                    "cell_view_mean_steps": float(cv.mean()),
                    "cell_view_sd_steps": float(cv.std(ddof=1)),
                    "cell_view_over_traditional_ratio": float(cv.mean() / tr.mean()),
                    "traditional_over_cell_view_ratio": float(tr.mean() / cv.mean()),
                    "paired_difference_mean_cell_minus_traditional": float(diff.mean()),
                    "paired_difference_sd": float(diff.std(ddof=1)),
                    "paired_difference_median": float(diff.median()),
                    "paired_difference_min": float(diff.min()),
                    "paired_difference_max": float(diff.max()),
                    "paper_relation": claim["paperRelation"],
                    "paper_reported_factor": paper_factor,
                    "observed_signed_direction": sign_direction,
                    "observed_practical_direction_5pct": practical,
                    "direction_matches_paper_sign_or_similarity": direction_matches_claim(claim, sign_direction, practical),
                    "observed_factor_using_paper_relation": float(factor) if factor is not None else None,
                    "factor_delta_from_paper": (
                        float(factor - paper_factor) if factor is not None and paper_factor is not None else None
                    ),
                    "paper_claim_text": claim["paperText"],
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["metric_id", "algorithm"]).reset_index(drop=True)


def validate_inputs_and_outputs(replicate_df: pd.DataFrame, condition_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []

    if len(replicate_df) != 600:
        errors.append(f"expected 600 S05 replicate rows from S04, found {len(replicate_df)}")
    grouped_counts = replicate_df.groupby(["algorithm", "implementation"]).size()
    if not grouped_counts.eq(100).all() or len(grouped_counts) != 6:
        errors.append(f"expected six algorithm/implementation groups with N=100, observed {grouped_counts.to_dict()}")
    if not replicate_df["completed"].all():
        errors.append("not all S04 source runs are marked completed")
    if not (replicate_df["final_sortedness_percent"] == 100.0).all():
        errors.append("not all S04 source runs reached final 100 percent Sortedness")
    if not (replicate_df["final_monotonicity_error"] == 0).all():
        errors.append("not all S04 source runs have zero final monotonicity error")

    pair_validation_rows = []
    for (algorithm, replicate_index), group in replicate_df.groupby(["algorithm", "replicate_index"], sort=True):
        implementations = sorted(set(group["implementation"]))
        seed_count = int(group["input_permutation_seed"].nunique())
        hash_count = int(group["initial_state_hash"].nunique())
        matched = set(implementations) == set(IMPLEMENTATIONS) and seed_count == 1 and hash_count == 1
        pair_validation_rows.append(
            {
                "algorithm": algorithm,
                "replicate_index": int(replicate_index),
                "implementation_count": len(implementations),
                "input_seed_unique_count": seed_count,
                "initial_hash_unique_count": hash_count,
                "matched": matched,
            }
        )
    matched_pair_count = sum(1 for row in pair_validation_rows if row["matched"])
    if matched_pair_count != 300:
        errors.append(f"expected 300 matched algorithm replicate pairs, found {matched_pair_count}")

    traditional = replicate_df[replicate_df["implementation"] == "traditional"]
    if not (traditional["comparison_only_count"] == traditional["raw_s04_comparison_count"]).all():
        errors.append("traditional comparison_only_count does not equal raw S04 comparison_count")
    if not (traditional["raw_s04_archived_compare_and_swap_count"] == 0).all():
        errors.append("traditional archived_compare_and_swap_count is unexpectedly nonzero")
    cell_view = replicate_df[replicate_df["implementation"] == "cell_view"]
    if not (cell_view["comparison_only_count"] == cell_view["raw_s04_archived_compare_and_swap_count"]).all():
        errors.append("cell-view comparison_only_count does not equal archived compare-and-swap count")
    if not (
        cell_view["raw_s04_comparison_count"]
        == cell_view["raw_s04_swap_count"] + cell_view["raw_s04_archived_compare_and_swap_count"]
    ).all():
        errors.append("cell-view raw S04 comparison_count is not swap_count plus archived compare-and-swap count")

    if len(condition_df) != 12:
        errors.append(f"expected 12 condition summary rows, found {len(condition_df)}")
    if len(pairwise_df) != 6:
        errors.append(f"expected 6 pairwise comparison rows, found {len(pairwise_df)}")
    if pairwise_df["pair_count"].min() != 100 or pairwise_df["pair_count"].max() != 100:
        errors.append("not every pairwise metric has N=100 matched pairs")

    success = not errors
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "passed" if success else "failed",
        "validationResult": (
            "passed: S05 used 600 completed S04 no-Frozen runs, six N=100 algorithm/implementation groups, "
            "300 matched algorithm pairs, 12 condition-metric rows, 6 paired-comparison rows, and consistent "
            "normalized comparison-count conventions."
            if success
            else "failed: " + "; ".join(errors)
        ),
        "errors": errors,
        "replicateRowCount": int(len(replicate_df)),
        "conditionMetricRowCount": int(len(condition_df)),
        "pairwiseRowCount": int(len(pairwise_df)),
        "algorithmImplementationGroupCounts": {
            f"{algorithm}/{implementation}": int(count)
            for (algorithm, implementation), count in grouped_counts.sort_index().items()
        },
        "matchedAlgorithmPairCount": int(matched_pair_count),
        "completedCount": int(replicate_df["completed"].sum()),
        "final100PercentSortednessCount": int((replicate_df["final_sortedness_percent"] == 100.0).sum()),
        "comparisonConventionChecks": {
            "traditionalComparisonOnlyEqualsRawComparisonCount": bool(
                (traditional["comparison_only_count"] == traditional["raw_s04_comparison_count"]).all()
            ),
            "traditionalArchivedCountAllZero": bool((traditional["raw_s04_archived_compare_and_swap_count"] == 0).all()),
            "cellViewComparisonOnlyEqualsArchivedCount": bool(
                (cell_view["comparison_only_count"] == cell_view["raw_s04_archived_compare_and_swap_count"]).all()
            ),
            "cellViewRawComparisonEqualsSwapPlusArchived": bool(
                (
                    cell_view["raw_s04_comparison_count"]
                    == cell_view["raw_s04_swap_count"] + cell_view["raw_s04_archived_compare_and_swap_count"]
                ).all()
            ),
        },
        "s06Started": False,
    }


def plot_figure04(condition_df: pd.DataFrame, pairwise_df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    colors = {"traditional": "#3b6ea8", "cell_view": "#c45733"}
    labels = {"traditional": "Traditional", "cell_view": "Cell-view"}
    metric_titles = {
        "swap_only_steps": "(a) Swaps only",
        "swap_plus_comparison_steps": "(b) Swaps plus comparisons",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=False)
    x = np.arange(len(ALGORITHMS))
    width = 0.34

    for ax, metric_id in zip(axes, METRIC_ORDER):
        metric_rows = condition_df[condition_df["metric_id"] == metric_id]
        for offset, implementation in [(-width / 2, "traditional"), (width / 2, "cell_view")]:
            subset = (
                metric_rows[metric_rows["implementation"] == implementation]
                .set_index("algorithm")
                .loc[ALGORITHMS]
                .reset_index()
            )
            ax.bar(
                x + offset,
                subset["mean_steps"],
                width,
                yerr=subset["sd_steps"],
                capsize=3,
                label=labels[implementation],
                color=colors[implementation],
                alpha=0.92,
                linewidth=0.7,
                edgecolor="#222222",
            )

        pair_rows = pairwise_df[pairwise_df["metric_id"] == metric_id].set_index("algorithm")
        ymax = float(metric_rows["mean_steps"].max() + metric_rows["sd_steps"].max())
        for i, algorithm in enumerate(ALGORITHMS):
            row = pair_rows.loc[algorithm]
            ratio = float(row["cell_view_over_traditional_ratio"])
            if metric_id == "swap_plus_comparison_steps" and algorithm in {"bubble", "insertion"}:
                ratio_text = f"T/CV {row['traditional_over_cell_view_ratio']:.2f}x"
            else:
                ratio_text = f"CV/T {ratio:.2f}x"
            ax.text(i, ymax * 1.05, ratio_text, ha="center", va="bottom", fontsize=9)

        ax.set_title(metric_titles[metric_id])
        ax.set_xticks(x)
        ax.set_xticklabels([name.title() for name in ALGORITHMS])
        ax.set_ylabel("Mean steps to sorted state")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_ylim(0, ymax * 1.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("E01 S05 Figure 4-style efficiency comparison, n=100, N=100", y=0.99)
    fig.text(0.5, 0.015, "Error bars show sample standard deviation across matched S04 replicates.", ha="center")
    fig.tight_layout(rect=[0, 0.04, 1, 0.89])
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_methods(
    path: Path,
    artifacts: list[str],
    validation_result: str,
    key_rows: list[list[Any]],
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    path.write_text(
        f"""# S05 Efficiency Methods

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_lines}
- Recommended next action: {recommended_next_action}

## Count Definitions

S05 uses the paper's two descriptive step definitions: swaps only, and swaps plus comparisons. The source S04 fields needed normalization before aggregation.

- `swap_only_steps`: the S04 `swap_count` field for both traditional and cell-view runs.
- `comparison_only_count`, traditional: the S04 `comparison_count` field from the S03/S04 wrapper implementation.
- `comparison_only_count`, cell-view: the S04 `archived_compare_and_swap_count` field, which is the archived `StatusProbe.compare_and_swap_count` code convention. This records cell-view `record_compare_and_swap()` calls when an archived cell policy reports that it should move; it is not an exhaustive count of every neighbor or target inspection.
- `swap_plus_comparison_steps`: `swap_only_steps + comparison_only_count` after the normalization above.

The raw S04 fields are preserved in `e01_efficiency_replicates.*` as `raw_s04_swap_count`, `raw_s04_comparison_count`, and `raw_s04_archived_compare_and_swap_count`.

## Paper-Claim Alignment

{markdown_table(["Metric", "Algorithm", "Traditional mean", "Cell-view mean", "Observed factor", "Paper claim", "Interpretation"], key_rows)}

No z-tests, p-values, bootstrap intervals, or effect-size inference were computed in S05; those are reserved for S06.
""",
        encoding="utf-8",
    )


def write_validation_md(path: Path, validation: dict[str, Any], artifacts: list[str], caveats: list[str], recommended_next_action: str) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    counts = [
        ["Replicate rows", validation["replicateRowCount"]],
        ["Condition metric rows", validation["conditionMetricRowCount"]],
        ["Pairwise rows", validation["pairwiseRowCount"]],
        ["Matched algorithm pairs", validation["matchedAlgorithmPairCount"]],
        ["Completed runs", validation["completedCount"]],
        ["Final 100 percent Sortedness runs", validation["final100PercentSortednessCount"]],
        ["S06 started", validation["s06Started"]],
    ]
    path.write_text(
        f"""# S05 Validation

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS if validation["success"] else "failed"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation["validationResult"]}
- Caveats or blockers:
{caveat_lines}
- Recommended next action: {recommended_next_action}

## Counts

{markdown_table(["Check", "Value"], counts)}

## Errors

{markdown_table(["Error"], [[error] for error in validation["errors"]] or [["None"]])}
""",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    artifacts: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    key_rows: list[list[Any]],
) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    path.write_text(
        f"""# S05 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S05 converted the completed S04 runs into efficiency tables for swaps only and swaps plus comparisons. Bubble and Insertion match exactly for swap-only counts, cell-view Selection needs many more swaps, and comparison-inclusive counts depend on the archived comparison-count convention documented in the methods note.
- Recommended next action: {recommended_next_action}

## Key Results

{markdown_table(["Metric", "Algorithm", "Traditional mean", "Cell-view mean", "Observed factor", "Paper claim", "Interpretation"], key_rows)}
""",
        encoding="utf-8",
    )


def key_result_rows(pairwise_df: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for metric_id in METRIC_ORDER:
        for algorithm in ALGORITHMS:
            row = pairwise_df[(pairwise_df["metric_id"] == metric_id) & (pairwise_df["algorithm"] == algorithm)].iloc[0]
            paper_factor = row["paper_reported_factor"]
            factor = row["observed_factor_using_paper_relation"]
            relation = row["paper_relation"]
            if relation == "similar":
                factor_text = f"CV/T {row['cell_view_over_traditional_ratio']:.2f}x"
            elif relation == "cell_view_fewer":
                factor_text = f"T/CV {factor:.2f}x"
            else:
                factor_text = f"CV/T {factor:.2f}x"
            if pd.notna(paper_factor):
                paper_text = f"{relation}; reported factor {paper_factor:g}x"
            else:
                paper_text = relation
            if row["direction_matches_paper_sign_or_similarity"]:
                interpretation = "direction matches, magnitude differs" if pd.notna(paper_factor) else "matches similar-swap claim"
                if pd.notna(paper_factor) and abs(float(row["factor_delta_from_paper"])) <= 0.25:
                    interpretation = "direction and rough magnitude match"
            else:
                interpretation = "does not match paper direction or similarity threshold"
            rows.append(
                [
                    "Swaps only" if metric_id == "swap_only_steps" else "Swaps plus comparisons",
                    str(algorithm).title(),
                    f"{row['traditional_mean_steps']:.2f}",
                    f"{row['cell_view_mean_steps']:.2f}",
                    factor_text,
                    paper_text,
                    interpretation,
                ]
            )
    return rows


def write_run_manifest(artifacts_dir: Path, status_payload: dict[str, Any], artifact_paths: list[Path], skip: bool) -> None:
    if skip:
        return
    provenance_dir = artifacts_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = provenance_dir / "run_manifest.json"
    manifest = load_json(run_manifest_path)
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["generatedAt"] = utc_now()
    manifest["researchStepId"] = STEP_ID
    manifest["recommendedNextAction"] = status_payload["recommendedNextAction"]
    manifest["caveatsOrBlockers"] = status_payload["caveatsOrBlockers"]
    manifest["artifactsWritten"] = status_payload["artifactsWritten"]
    manifest["git"] = status_payload["git"]
    manifest["runtime"] = status_payload["runtime"]
    manifest.setdefault("researchSteps", {})
    artifacts = collect_artifacts(artifact_paths)
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "validationResult": status_payload["validationResult"],
        "outcomeClassification": status_payload["outcomeClassification"],
        "generatedAt": status_payload["generatedAt"],
    }
    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_s05(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e01"
    for directory in [step_dir, code_dir, results_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    s04_path = Path(args.s04_replicate_summary)
    if not s04_path.exists():
        raise RuntimeError(f"missing required S04 replicate summary: {s04_path}")

    s04_df = pd.read_parquet(s04_path)
    replicate_df = normalize_replicates(s04_df)
    condition_df = summarize_conditions(replicate_df)
    pairwise_df = summarize_pairwise(replicate_df)
    validation = validate_inputs_and_outputs(replicate_df, condition_df, pairwise_df)

    efficiency_replicates_parquet = results_dir / "e01_efficiency_replicates.parquet"
    efficiency_replicates_csv = results_dir / "e01_efficiency_replicates.csv"
    efficiency_parquet = results_dir / "e01_efficiency.parquet"
    efficiency_csv = results_dir / "e01_efficiency.csv"
    pairwise_parquet = results_dir / "e01_efficiency_pairwise.parquet"
    pairwise_csv = results_dir / "e01_efficiency_pairwise.csv"
    e01_results_parquet = results_dir / "e01_results.parquet"
    e01_results_csv = results_dir / "e01_results.csv"
    figure_png = figure_dir / "figure04_efficiency.png"
    figure_pdf = figure_dir / "figure04_efficiency.pdf"
    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    methods_md = step_dir / "efficiency_methods.md"
    summary_md = step_dir / "summary.md"
    status_json = step_dir / "status.json"
    artifact_manifest_json = step_dir / "artifact_manifest.json"
    code_copy = code_dir / Path(__file__).name

    replicate_df.to_parquet(efficiency_replicates_parquet, index=False)
    replicate_df.to_csv(efficiency_replicates_csv, index=False)
    condition_df.to_parquet(efficiency_parquet, index=False)
    condition_df.to_csv(efficiency_csv, index=False)
    pairwise_df.to_parquet(pairwise_parquet, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)
    condition_df.assign(result_scope="S05_efficiency").to_parquet(e01_results_parquet, index=False)
    condition_df.assign(result_scope="S05_efficiency").to_csv(e01_results_csv, index=False)
    plot_figure04(condition_df, pairwise_df, figure_png, figure_pdf)
    shutil.copy2(Path(__file__), code_copy)

    recommended_next_action = (
        "Hand control back to the Chief Scientist workflow; proceed to S06 only after instruction, using S05 "
        "efficiency tables for reported z-tests and secondary intervals."
    )
    caveats = [
        "S05 reuses S04 deterministic no-Frozen runs rather than rerunning the original OS-threaded archive.",
        "Traditional algorithms are S03/S04 wrapper implementations because S02 found no clean original traditional runners.",
        "Cell-view comparison-only counts use the archived StatusProbe.compare_and_swap_count convention, which counts should-move compare-and-swap opportunities rather than every neighbor or target inspection.",
        "S04 stored cell-view comparison_count as archived compare-and-swap count plus swaps, while traditional comparison_count was comparisons only; S05 normalizes these into explicit fields before aggregation.",
        "Bubble and Insertion swap-only counts are identical across implementations under the S04 matched-input wrappers; this is expected from the inversion-count-equivalent swap behavior.",
        "S05 reports descriptive means, standard deviations, ratios, and paired differences only; z-tests and confidence intervals are reserved for S06.",
        "No Frozen Cell, Delayed Gratification, Aggregation, or S06 statistical analyses were started.",
    ]
    key_rows = key_result_rows(pairwise_df)

    artifact_paths = [
        efficiency_replicates_parquet,
        efficiency_replicates_csv,
        efficiency_parquet,
        efficiency_csv,
        pairwise_parquet,
        pairwise_csv,
        e01_results_parquet,
        e01_results_csv,
        figure_png,
        figure_pdf,
        validation_json,
        validation_md,
        methods_md,
        summary_md,
        status_json,
        artifact_manifest_json,
        code_copy,
    ]
    artifacts_written = [str(path) for path in artifact_paths]

    validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_validation_md(validation_md, validation, artifacts_written, caveats, recommended_next_action)
    write_methods(methods_md, artifacts_written, validation["validationResult"], key_rows, caveats, recommended_next_action)
    write_summary(summary_md, artifacts_written, validation["validationResult"], caveats, recommended_next_action, key_rows)

    s04_status = load_json(Path(args.s04_status))
    config = load_json(Path(args.config))
    git_meta = get_git_metadata()
    runtime = {
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "osCpuCount": os.cpu_count(),
        "workerCount": 1,
        "gpuUsed": False,
        "matplotlibBackend": matplotlib.get_backend(),
        "numpyVersion": np.__version__,
        "pandasVersion": pd.__version__,
    }
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": STATUS if validation["success"] else "failed",
        "artifactsWritten": artifacts_written,
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": caveats + validation["errors"],
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation["success"] else "constraining",
        "generatedAt": utc_now(),
        "experimentId": EXPERIMENT_ID,
        "sourceArtifacts": {
            "s04ReplicateSummary": str(s04_path),
            "s04Status": str(args.s04_status),
            "s03Config": str(args.config),
            "s03SeedTable": str(args.seed_table),
        },
        "sourceS04": {
            "researchStepId": s04_status.get("researchStepId"),
            "success": s04_status.get("success"),
            "status": s04_status.get("status"),
            "validationResult": s04_status.get("validationResult"),
            "commit": s04_status.get("git", {}).get("commit"),
        },
        "frozenConfig": {
            "repeatCount": config.get("paperBaseline", {}).get("repeatCount"),
            "arrayLength": config.get("paperBaseline", {}).get("uniqueInputProfile", {}).get("n"),
            "inputProfile": "unique_1_100",
            "frozenCellCount": 0,
        },
        "rowCounts": {
            "replicateRows": int(len(replicate_df)),
            "conditionMetricRows": int(len(condition_df)),
            "pairwiseRows": int(len(pairwise_df)),
        },
        "paperClaimAlignment": pairwise_df[
            [
                "metric_id",
                "algorithm",
                "paper_relation",
                "paper_reported_factor",
                "observed_factor_using_paper_relation",
                "direction_matches_paper_sign_or_similarity",
            ]
        ].to_dict(orient="records"),
        "s06Started": False,
        "git": git_meta,
        "runtime": runtime,
    }
    status_json.write_text(
        json.dumps(json_ready(status_payload), indent=2, sort_keys=True, allow_nan=False, default=as_builtin) + "\n",
        encoding="utf-8",
    )

    final_artifact_paths = [path for path in artifact_paths if path.exists() and path != artifact_manifest_json]
    artifact_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "status": STATUS if validation["success"] else "failed",
        "success": bool(validation["success"]),
        "validationResult": validation["validationResult"],
        "artifacts": collect_artifacts(final_artifact_paths),
        "sourceArtifacts": status_payload["sourceArtifacts"],
    }
    artifact_manifest_json.write_text(
        json.dumps(json_ready(artifact_manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final_artifact_paths.append(artifact_manifest_json)
    write_run_manifest(artifacts_dir, status_payload, final_artifact_paths, args.skip_manifest_update)

    print(validation["validationResult"])
    print(f"Wrote {len(final_artifact_paths)} S05 artifact files under {artifacts_dir}")
    return 0 if validation["success"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--s04-replicate-summary", default=str(S04_REPLICATE_SUMMARY_DEFAULT))
    parser.add_argument("--s04-status", default=str(S04_STATUS_DEFAULT))
    parser.add_argument("--config", default=str(S03_CONFIG_DEFAULT))
    parser.add_argument("--seed-table", default=str(S03_SEED_TABLE_DEFAULT))
    parser.add_argument("--skip-manifest-update", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_s05(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
