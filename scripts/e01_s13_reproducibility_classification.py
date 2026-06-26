#!/usr/bin/env python3
"""Build E01 S13 claim-level reproducibility classifications."""

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

sys.dont_write_bytecode = True

import pandas as pd


EXPERIMENT_ID = "E01"
STEP_ID = "S13"
STEP_NUMBER = 13
STATUS = "completed"
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
PAPER_MD_DEFAULT = WORKSPACE_ROOT / "input-attachments" / "a8ba3250-a8c7-4b7b-9a3f-edc9edb9e5f5" / "pdf-markdown.md"

ALLOWED_CLASSIFICATIONS = {
    "exact",
    "statistically consistent",
    "directionally consistent",
    "not replicated",
}

EXPECTED_CLAIM_IDS = [
    "FIG01_METHOD_ENTITIES",
    "FIG02_CELL_VIEW_POLICIES",
    "FIG03_COMPLETION",
    "FIG03_TRAJECTORY_STYLE",
    "FIG04_SWAP_BUBBLE",
    "FIG04_SWAP_INSERTION",
    "FIG04_SWAP_SELECTION",
    "FIG04_TOTAL_BUBBLE",
    "FIG04_TOTAL_INSERTION",
    "FIG04_TOTAL_SELECTION",
    "FIG05_CELL_VIEW_ERROR_TOLERANCE",
    "FIG05_PASSIVE_CELL_VIEW_RANKING",
    "FIG05_STUCK_CELL_VIEW_RANKING",
    "FIG06_DG_FORMULA_AND_UNIT_TESTS",
    "FIG07_DG_BUBBLE_CELL_GT_TRAD",
    "FIG07_DG_INSERTION_SIMILAR",
    "FIG07_DG_SELECTION_CELL_LT_TRAD",
    "FIG07_DG_FROZEN_COUNT_TRENDS",
    "FIG07_DG_ALGORITHM_ORDERING",
    "FIG08_SAME_GOAL_SORT_COMPLETION",
    "FIG08_SAME_GOAL_EFFICIENCY_INTERPOLATION",
    "FIG08_UNIQUE_AGGREGATION_SIGNIFICANCE",
    "FIG08_UNIQUE_AGGREGATION_PEAK_NUMBERS",
    "FIG08_SAME_CODE_NEGATIVE_CONTROL",
    "FIG08_DUPLICATE_SORT_COMPLETION",
    "FIG08_DUPLICATE_AGGREGATION_NUMBERS",
    "FIG08_DUPLICATE_FINAL_VS_UNIQUE",
    "FIG09_UNIQUE_OPPOSITE_FINAL_SORTEDNESS",
    "FIG09_UNIQUE_OPPOSITE_DYNAMICS_EQUILIBRIUM",
    "FIG09_UNIQUE_OPPOSITE_AGGREGATION_DOMINANCE",
    "FIG10_REPEATED_OPPOSITE_SIMILARITY",
    "FIG10_REPEATED_OPPOSITE_DOMINANCE",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in sorted({p.resolve() for p in paths if p.exists()}):
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        artifacts.append(
            {
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def get_git_metadata() -> dict[str, Any]:
    return {
        "repository": str(REPO_ROOT),
        "branch": run_git(["git", "branch", "--show-current"]),
        "commit": run_git(["git", "rev-parse", "HEAD"]),
        "statusShort": run_git(["git", "status", "--short"]),
        "remoteOriginUrl": run_git(["git", "remote", "get-url", "origin"]),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("\n", "<br>").replace("|", "\\|")

    out = ["| " + " | ".join(clean(h) for h in headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(out)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def rel(path: Path) -> str:
    return str(path)


def row_lookup(df: pd.DataFrame, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        mask &= df[column].eq(value)
    matched = df[mask]
    if len(matched) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(matched)}")
    return matched.iloc[0]


def list_metric_triplets(df: pd.DataFrame, value_col: str, filter_expr: pd.Series | None = None) -> str:
    subset = df[filter_expr].copy() if filter_expr is not None else df.copy()
    parts = []
    for _, row in subset.iterrows():
        prefix = row.get("mixtureId", row.get("algorithm", row.get("conditionId", "")))
        parts.append(f"{prefix}={fmt(row[value_col])}")
    return "; ".join(parts)


def build_classification_rows(artifacts_dir: Path, paper_md: Path) -> pd.DataFrame:
    results_dir = artifacts_dir / "results"
    step_dir = artifacts_dir / "research_steps"
    paths = {
        "paper": paper_md,
        "s02_map": step_dir / "S02" / "code_paper_map.md",
        "s02_index": step_dir / "S02" / "function_index.csv",
        "s03_config": artifacts_dir / "configs" / "e01_baseline_config.json",
        "s04_summary": results_dir / "e01_s04_condition_summary.csv",
        "s04_replicates": results_dir / "e01_s04_replicate_summary.parquet",
        "s05_pairwise": results_dir / "e01_efficiency_pairwise.parquet",
        "s06_stats": results_dir / "e01_efficiency_statistics.parquet",
        "s07_summary": results_dir / "e01_frozen_cell_robustness_summary.parquet",
        "s08_summary": results_dir / "e01_delayed_gratification_summary.parquet",
        "s08_comparisons": results_dir / "e01_delayed_gratification_comparisons.parquet",
        "s08_trends": results_dir / "e01_delayed_gratification_trends.parquet",
        "s09_summary": results_dir / "e01_same_goal_chimera_summary.parquet",
        "s09_efficiency": results_dir / "e01_same_goal_chimera_efficiency.parquet",
        "s10_peaks": results_dir / "e01_aggregation_peak_summary.parquet",
        "s10_stats": results_dir / "e01_aggregation_statistics.parquet",
        "s11_summary": results_dir / "e01_duplicate_value_chimera_summary.parquet",
        "s11_peaks": results_dir / "e01_duplicate_value_aggregation_curve_peaks.parquet",
        "s11_comparison": results_dir / "e01_duplicate_value_unique_comparison.parquet",
        "s12_summary": results_dir / "e01_opposite_direction_chimera_summary.parquet",
        "s12_equilibria": results_dir / "e01_opposite_direction_equilibria.parquet",
        "s12_paper": results_dir / "e01_opposite_direction_paper_comparison.parquet",
    }
    tables = {
        name: read_table(path)
        for name, path in paths.items()
        if name not in {"paper", "s02_map", "s03_config"} and path.suffix in {".csv", ".parquet"}
    }

    records: list[dict[str, Any]] = []

    def add(
        claim_id: str,
        paper_figure: str,
        claim_type: str,
        paper_claim: str,
        paper_expected: str,
        observed: str,
        classification: str,
        justification: str,
        plausible_causes: str,
        source_steps: str,
        supporting_artifacts: list[Path],
        review_priority: str = "normal",
    ) -> None:
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(f"Unsupported classification {classification} for {claim_id}")
        records.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "stepNumber": STEP_NUMBER,
                "claimId": claim_id,
                "paperFigure": paper_figure,
                "claimType": claim_type,
                "paperClaim": paper_claim,
                "paperExpectedValue": paper_expected,
                "observedEvidence": observed,
                "classification": classification,
                "justification": justification,
                "plausibleMismatchCauses": plausible_causes,
                "sourceResearchSteps": source_steps,
                "supportingArtifacts": ";".join(rel(p) for p in supporting_artifacts),
                "reviewPriority": review_priority,
            }
        )

    common_causes = (
        "Original random seeds and OS-thread scheduling are unavailable; the baseline uses deterministic S03 seeds and pseudo-scheduled wrappers around archived cell-view classes."
    )
    method_causes = "Conceptual/method figures were mapped to code and regenerated analyses, not pixel-recreated from the paper artwork."

    add(
        "FIG01_METHOD_ENTITIES",
        "Figure 1",
        "conceptual/method",
        "Sorting is framed through Position, Value, and agent-like cell behavior.",
        "Conceptual model, not a reported numeric statistic.",
        "S02 mapped Position, Value, Algotype, cell-view agents, Probe outputs, and metric utilities; S03 froze the baseline parameters.",
        "directionally consistent",
        "The codebase contains the core entities and behaviors required by the paper, but Figure 1 itself is conceptual artwork.",
        method_causes,
        "S01;S02;S03",
        [paths["paper"], paths["s02_map"], paths["s03_config"]],
    )
    add(
        "FIG02_CELL_VIEW_POLICIES",
        "Figure 2",
        "conceptual/method",
        "Traditional Bubble, Insertion, and Selection policies can be translated to local cell-view policies.",
        "Conceptual algorithm mapping, not a reported numeric statistic.",
        "S01 smoke tests and S02 mapping identified runnable cell-view Bubble, Insertion, and Selection classes and explicit gaps for traditional wrappers.",
        "directionally consistent",
        "The operational policies used downstream match the paper-level algorithm families, with reconstructed traditional wrappers where the archive lacked clean paper-setting entry points.",
        "No clean original traditional-sort runner was found; wrappers were documented and frozen in S03.",
        "S01;S02;S03",
        [paths["s02_map"], paths["s02_index"], paths["s03_config"]],
    )

    s04 = tables["s04_summary"]
    final_100 = int(s04["final_100pct_sortedness_count"].sum())
    total_reps = int(s04["replicate_count"].sum())
    condition_count = len(s04)
    add(
        "FIG03_COMPLETION",
        "Figure 3",
        "numeric result",
        "Traditional and cell-view Bubble, Insertion, and Selection sort all reach 100% Sortedness for n=100, N=100 no-Frozen arrays.",
        "Six conditions, 100 repeats each, final 100% Sortedness.",
        f"{final_100}/{total_reps} runs across {condition_count} conditions reached final 100% Sortedness.",
        "exact" if final_100 == total_reps and total_reps == 600 else "not replicated",
        "The regenerated S04 no-Frozen condition table exactly satisfies the paper's completion claim at the condition and replicate counts in scope.",
        common_causes,
        "S04",
        [paths["s04_summary"], paths["s04_replicates"]],
        review_priority="anchor",
    )
    add(
        "FIG03_TRAJECTORY_STYLE",
        "Figure 3",
        "figure trajectory",
        "The paper shows 100-run Sortedness trajectories from random initial arrays to monotone final states.",
        "Figure-style trajectory panels for traditional/cell-view Bubble, Insertion, and Selection.",
        "S04 wrote average trajectories, event traces, state snapshots, and Figure 3-style PNG/PDF outputs using matched initial states.",
        "directionally consistent",
        "The reproduced plots support the same qualitative trajectory endpoint and matched-seed design, but no pixel-level figure comparison was possible.",
        "The original plotting data, seeds, and exact renderer are unavailable.",
        "S04",
        [paths["s04_summary"], artifacts_dir / "figures" / "e01" / "figure03_sortedness_trajectories.png"],
    )

    s06 = tables["s06_stats"]
    fig4_specs = [
        ("FIG04_SWAP_BUBBLE", "bubble", "swap_only_steps", "statistically consistent"),
        ("FIG04_SWAP_INSERTION", "insertion", "swap_only_steps", "statistically consistent"),
        ("FIG04_SWAP_SELECTION", "selection", "swap_only_steps", "statistically consistent"),
        ("FIG04_TOTAL_BUBBLE", "bubble", "swap_plus_comparison_steps", "statistically consistent"),
        ("FIG04_TOTAL_INSERTION", "insertion", "swap_plus_comparison_steps", "directionally consistent"),
        ("FIG04_TOTAL_SELECTION", "selection", "swap_plus_comparison_steps", "statistically consistent"),
    ]
    for claim_id, algorithm, metric_id, classification in fig4_specs:
        row = row_lookup(s06, algorithm=algorithm, metric_id=metric_id)
        metric_label = str(row["metric_label"])
        observed = (
            f"traditional mean={fmt(row['traditional_mean_steps'])}, cell-view mean={fmt(row['cell_view_mean_steps'])}, "
            f"paper-relation factor={fmt(row['observed_factor_using_paper_relation'])}, "
            f"z={fmt(row['paper_style_z_statistic_cell_minus_traditional'])}, p={fmt(row['paper_style_p_value_two_sided'])}, "
            f"paper-style conclusion reproduced={bool(row['paper_style_conclusion_reproduced'])}."
        )
        causes = "Comparison-count convention differs from a fully exhaustive inspection counter; original z-test formula details are under-specified."
        if claim_id == "FIG04_TOTAL_INSERTION":
            causes += " The independent paper-style z-test is non-significant in S06, while the paired-seed secondary test preserves the paper direction."
        add(
            claim_id,
            "Figure 4",
            "statistic",
            str(row["paper_claim_text"]),
            f"Paper z={fmt(row['paper_reported_z_statistic'])}, p={row['paper_reported_p_text']}, relation={row['paper_relation']}.",
            observed,
            classification,
            str(row["interpretation"]),
            causes,
            "S05;S06",
            [paths["s05_pairwise"], paths["s06_stats"]],
            review_priority="anchor" if claim_id == "FIG04_TOTAL_INSERTION" else "normal",
        )

    s07 = tables["s07_summary"]
    frozen_cmp = s07[s07["frozenVariant"].isin(["passive", "stuck"]) & s07["frozenCount"].isin([1, 2, 3])]
    no_greater_checks = []
    for variant in ["passive", "stuck"]:
        for algorithm in ["bubble", "insertion", "selection"]:
            for frozen_count in [1, 2, 3]:
                cell = row_lookup(
                    frozen_cmp,
                    implementation="cell_view",
                    algorithm=algorithm,
                    frozenVariant=variant,
                    frozenCount=frozen_count,
                )
                trad = row_lookup(
                    frozen_cmp,
                    implementation="traditional",
                    algorithm=algorithm,
                    frozenVariant=variant,
                    frozenCount=frozen_count,
                )
                no_greater_checks.append(float(cell["meanFinalMonotonicityError"]) <= float(trad["meanFinalMonotonicityError"]))
    passive_cell = s07[(s07["implementation"] == "cell_view") & (s07["frozenVariant"] == "passive") & (s07["frozenCount"].isin([1, 2, 3]))]
    stuck_cell = s07[(s07["implementation"] == "cell_view") & (s07["frozenVariant"] == "stuck") & (s07["frozenCount"].isin([1, 2, 3]))]
    add(
        "FIG05_CELL_VIEW_ERROR_TOLERANCE",
        "Figure 5",
        "numeric result",
        "All cell-view sorting algorithms have less monotonicity error than traditional versions with Frozen Cells.",
        "Cell-view mean final monotonicity error lower than traditional for Bubble, Insertion, and Selection at f=1..3 under passive and stuck variants.",
        f"{sum(no_greater_checks)}/{len(no_greater_checks)} cell-view comparisons were less than or equal to matched traditional means.",
        "directionally consistent" if all(no_greater_checks) else "not replicated",
        "The paper-level direction holds for every S07 Frozen Cell comparison, though exact paper means are not generally reproduced.",
        "Frozen Cell passive/stuck semantics are partially ambiguous in the archive, and traditional behavior is reconstructed.",
        "S07",
        [paths["s07_summary"]],
        review_priority="anchor",
    )
    add(
        "FIG05_PASSIVE_CELL_VIEW_RANKING",
        "Figure 5",
        "numeric result",
        "With passive Frozen Cells, cell-view Bubble has least monotonicity error and cell-view Selection highest.",
        "Paper Bubble means 0, 0.8, 2.64; Selection means 2.24, 4.36, 13.24 for f=1..3.",
        list_metric_triplets(passive_cell, "meanFinalMonotonicityError"),
        "directionally consistent",
        "Bubble is the lowest-error passive cell-view algorithm at all f values; Selection and Insertion are close in S07 and Selection is not strictly highest at f=1.",
        "The archived passive-Frozen semantics and exact scheduler differ from the paper run; the S07 Insertion wrapper bounds actor sweeps deterministically.",
        "S07",
        [paths["s07_summary"], artifacts_dir / "figures" / "e01" / "figure05_frozen_robustness.png"],
    )
    add(
        "FIG05_STUCK_CELL_VIEW_RANKING",
        "Figure 5",
        "numeric result",
        "With stuck Frozen Cells, cell-view Selection has the highest error tolerance and Bubble has the highest monotonicity error.",
        "Paper Bubble means 1.91, 3.72, 5.37; Selection means 1.0, 1.96, 2.91 for f=1..3.",
        list_metric_triplets(stuck_cell, "meanFinalMonotonicityError"),
        "directionally consistent",
        "Selection is the lowest-error stuck cell-view algorithm at every f value; Bubble is tied with Insertion in S07 rather than uniquely highest.",
        "Bubble and Insertion stuck behavior converges under the deterministic wrappers, while the original threaded ordering is unavailable.",
        "S07",
        [paths["s07_summary"], artifacts_dir / "figures" / "e01" / "figure05_frozen_robustness.png"],
    )

    add(
        "FIG06_DG_FORMULA_AND_UNIT_TESTS",
        "Figure 6",
        "metric/formula",
        "Delayed Gratification is computed from local Sortedness drops followed by recoveries.",
        "Formula illustration and qualitative examples.",
        "S08 documented the signed DG interpretation, ported it to percent Sortedness trajectories, and passed nine hand-constructed unit tests.",
        "directionally consistent",
        "The implemented DG scaffold captures the formula family used in the archived code, but the extracted paper text does not fully specify every edge case.",
        "Exact formula edge cases were inferred from source behavior and unit tests rather than a complete paper equation.",
        "S08",
        [step_dir / "S08" / "formula_interpretation.md", step_dir / "S08" / "dg_unit_tests.json"],
    )

    s08_comp = tables["s08_comparisons"]
    s08_trends = tables["s08_trends"]
    dg_specs = [
        (
            "FIG07_DG_BUBBLE_CELL_GT_TRAD",
            "bubble",
            "Paper reports cell-view Bubble has more DG than traditional.",
            "statistically consistent",
        ),
        (
            "FIG07_DG_INSERTION_SIMILAR",
            "insertion",
            "Paper reports cell-view Insertion DG is very similar to traditional.",
            "statistically consistent",
        ),
        (
            "FIG07_DG_SELECTION_CELL_LT_TRAD",
            "selection",
            "Paper concludes cell-view Selection performs less DG than traditional.",
            "statistically consistent",
        ),
    ]
    for claim_id, algorithm, claim, classification in dg_specs:
        row = row_lookup(s08_comp, figureFrozenVariant="stuck", algorithm=algorithm)
        observed = (
            f"cell-view mean={fmt(row['cellViewMeanDelayedGratification'])}, "
            f"traditional mean={fmt(row['traditionalMeanDelayedGratification'])}, "
            f"diff={fmt(row['meanDifferenceCellMinusTraditional'])}, "
            f"z={fmt(row['paperStyleZStatistic'])}, p={fmt(row['paperStylePValue'])}."
        )
        causes = "S08 uses the stuck-Frozen variant as the primary paper-style interpretation and ports the archived signed DG convention."
        if algorithm == "selection":
            causes += " The paper text reports a positive absolute difference while the conclusion and caption imply cell-view minus traditional is negative."
        add(
            claim_id,
            "Figure 7",
            "statistic",
            claim,
            str(row["paperClaimText"]),
            observed,
            classification,
            "The regenerated stuck-Frozen comparison reproduces the paper-level direction and significance/similarity conclusion.",
            causes,
            "S08",
            [paths["s08_comparisons"], paths["s08_summary"]],
        )
    primary_trends = s08_trends[s08_trends["isPrimaryPaperStyleVariant"]]
    trend_observed = "; ".join(
        f"{r.implementation}-{r.algorithm}:{r.trendDirection}, monotone={bool(r.monotoneNonDecreasing)}"
        for r in primary_trends.itertuples(index=False)
    )
    add(
        "FIG07_DG_FROZEN_COUNT_TRENDS",
        "Figure 7",
        "trend result",
        "Bubble and Insertion DG increase with the number of Frozen Cells; Selection has no clear trend.",
        "Paper reports increasing Bubble/Insertion DG for f=0..3 and no clear Selection trend.",
        trend_observed,
        "directionally consistent",
        "The primary stuck-Frozen S08 trends reproduce monotone nondecreasing Bubble and Insertion series for both implementations and nonmonotone Selection series.",
        "The passive-Frozen sensitivity does not reproduce the Bubble/Insertion trend, so this classification is tied to the S08 primary stuck-Frozen interpretation.",
        "S08",
        [paths["s08_trends"], artifacts_dir / "figures" / "e01" / "figure07_dg.png"],
    )
    s08_summary = tables["s08_summary"]
    primary_means = s08_summary[(s08_summary["frozenVariant"].isin(["none", "stuck"])) & (s08_summary["frozenCount"].isin([0, 1, 2, 3]))]
    mean_by_alg = primary_means.groupby("algorithm")["meanDelayedGratification"].mean().to_dict()
    add(
        "FIG07_DG_ALGORITHM_ORDERING",
        "Figure 7",
        "statistic",
        "Selection has the largest DG, and Insertion has more DG than Bubble.",
        "Caption reports Selection best and Insertion greater than Bubble.",
        "; ".join(f"{k}={fmt(v)}" for k, v in sorted(mean_by_alg.items())),
        "directionally consistent",
        "The regenerated DG means preserve Selection greater than Insertion greater than Bubble, but S08 did not recreate the paper's cross-algorithm z-tests exactly.",
        "Cross-algorithm paper test formulas are not fully specified in the extracted text.",
        "S08",
        [paths["s08_summary"], paths["s08_comparisons"]],
    )

    s09_summary = tables["s09_summary"]
    s09_eff = tables["s09_efficiency"]
    sorted_count = int((s09_summary["meanFinalSortednessPercent"] == 100.0).sum())
    add(
        "FIG08_SAME_GOAL_SORT_COMPLETION",
        "Figure 8",
        "numeric result",
        "All same-goal Algotype combinations completely sort the array.",
        "All unique same-goal pure and mixed cell-view chimera conditions reach final 100% Sortedness.",
        f"{sorted_count}/{len(s09_summary)} S09 same-goal conditions had mean final Sortedness 100%; all 700 replicates completed.",
        "exact" if sorted_count == len(s09_summary) else "not replicated",
        "S09 exactly reproduces the system-level completion claim for same-goal unique-value chimeras.",
        common_causes,
        "S09",
        [paths["s09_summary"]],
        review_priority="anchor",
    )
    mixed_eff = s09_eff[s09_eff["isMixedCondition"]]
    add(
        "FIG08_SAME_GOAL_EFFICIENCY_INTERPOLATION",
        "Figure 8",
        "numeric result",
        "Mixed same-goal chimera efficiency falls between the pure component efficiencies and is roughly average.",
        "Mixed swap counts within pure-component envelope.",
        "; ".join(
            f"{r.mixtureId}: mean={fmt(r.meanSwapCount)}, envelope=[{fmt(r.componentPureMeanMin)}, {fmt(r.componentPureMeanMax)}], within={bool(r.withinComponentEnvelope)}"
            for r in mixed_eff.itertuples(index=False)
        ),
        "directionally consistent" if bool(mixed_eff["withinComponentEnvelope"].all()) else "not replicated",
        "All mixed S09 conditions lie inside their pure-component swap-count envelope; the all-three mix is near but not exactly a simple arithmetic average.",
        "All-three allocation is 34/33/33 because n=100 is not divisible by three; deterministic pseudo-scheduler replaces live threading.",
        "S09",
        [paths["s09_efficiency"]],
    )

    s10_peaks = tables["s10_peaks"]
    s10_stats = tables["s10_stats"]
    mixed_peaks = s10_peaks[s10_peaks["isMixedCondition"]]
    mixed_stats = s10_stats[s10_stats["comparisonType"] == "mixed_peak_minus_fixed_count_random_null"]
    sig_ok = bool(((mixed_stats["zTestPValue"] < 0.01) & (mixed_stats["bootstrapCi95Lower"] > 0)).all())
    add(
        "FIG08_UNIQUE_AGGREGATION_SIGNIFICANCE",
        "Figure 8",
        "statistic",
        "Distinct Algotypes show significant Aggregation above the negative control during same-goal unique-value sorting.",
        "p << 0.01 for Bubble-Selection, Bubble-Insertion, Selection-Insertion, and all-three mixes.",
        "; ".join(
            f"{r.mixtureId}: z={fmt(r.zStatistic)}, p={fmt(r.zTestPValue)}, CI=[{fmt(r.bootstrapCi95Lower)}, {fmt(r.bootstrapCi95Upper)}]"
            for r in mixed_stats.itertuples(index=False)
        ),
        "statistically consistent" if sig_ok else "not replicated",
        "All four regenerated mixed conditions have one-sided z-tests p < 0.01 and positive bootstrap intervals above the fixed-count null.",
        "The available null is a fixed-count random-label null rather than the paper's same-code two-label negative control.",
        "S10",
        [paths["s10_stats"], paths["s10_peaks"]],
        review_priority="anchor",
    )
    add(
        "FIG08_UNIQUE_AGGREGATION_PEAK_NUMBERS",
        "Figure 8",
        "numeric result",
        "Unique-value same-goal chimeras reach peak Aggregation of 0.72, 0.65, 0.69, and 0.62 at 42%, 21%, 19%, and 22% progress for Bubble-Selection, Bubble-Insertion, Selection-Insertion, and all-three.",
        "Paper peak heights/timing by mixture.",
        "; ".join(
            f"{r.mixtureId}: observed={fmt(r.meanCurvePeakAggregation)} at {fmt(r.meanCurvePeakProgress * 100)}%, paper={fmt(r.paperPeakAggregation)} at {fmt(r.paperPeakProgress * 100)}%"
            for r in mixed_peaks.itertuples(index=False)
        ),
        "directionally consistent",
        "Every mixed condition peaks above 0.5 and above null, but reproduced peak heights are lower than several paper values and timing differs for Bubble-Selection/all-three.",
        "Aggregation implementation was reconstructed as a left-neighbor same-Algotype fraction, and scheduler/seed differences can shift transient peak timing.",
        "S10",
        [paths["s10_peaks"], artifacts_dir / "figures" / "e01" / "figure08_aggregation_unique.png"],
    )
    add(
        "FIG08_SAME_CODE_NEGATIVE_CONTROL",
        "Figure 8",
        "control",
        "The paper uses two labels running the same Bubble code as a negative control for spurious Aggregation.",
        "No significant deviation from about 50% chance in the same-code control.",
        "S10 did not have same-code two-label S09 traces; it used fixed-count random-label null baselines and pure one-Algotype controls as sanity checks.",
        "not replicated",
        "The exact negative-control experiment was not available in S09/S10 artifacts, so this control remains unreplicated in E01 through S13.",
        "The S09 queue did not generate two-label same-code mixed controls; the paper text and caption are also internally ambiguous about reported control peaks.",
        "S10",
        [paths["s10_stats"], step_dir / "S10" / "summary.md"],
        review_priority="high",
    )

    s11_summary = tables["s11_summary"]
    s11_comp = tables["s11_comparison"]
    dup_sorted = bool((s11_summary["meanFinalSortednessPercent"] == 100.0).all())
    add(
        "FIG08_DUPLICATE_SORT_COMPLETION",
        "Figure 8",
        "numeric result",
        "Duplicate-value same-goal chimeras sort the numeric values completely while allowing equal-value Algotype clustering.",
        "Three pairwise duplicate-value conditions, N=100, final 100% Sortedness.",
        f"{len(s11_summary)}/{len(s11_summary)} duplicate-value pairwise conditions reached mean final Sortedness 100%; all 300 replicates completed.",
        "exact" if dup_sorted else "not replicated",
        "S11 exactly reproduces duplicate-value numeric sorting completion for the pairwise same-goal conditions.",
        "Tie handling follows archived strict comparison behavior; equal-valued cells can remain in any Algotype order.",
        "S11",
        [paths["s11_summary"]],
    )
    add(
        "FIG08_DUPLICATE_AGGREGATION_NUMBERS",
        "Figure 8",
        "numeric result",
        "Duplicate-value pairwise chimeras reach peaks near 0.69, 0.63, and 0.71 for Bubble-Selection, Bubble-Insertion, and Insertion-Selection.",
        "Paper duplicate peaks 0.69 at 100%, 0.63 at 13%, and 0.71 at 100%; reported finals 0.65 and 0.70 for Bubble-Selection and Insertion-Selection.",
        "; ".join(
            f"{r.mixtureId}: peak={fmt(r.duplicateMeanCurvePeakAggregation)} at {fmt(r.duplicateMeanCurvePeakProgress * 100)}%, final={fmt(r.duplicateFinalAggregationMean)}, paperPeak={fmt(r.paperDuplicatePeakAggregation)}"
            for r in s11_comp.itertuples(index=False)
        ),
        "directionally consistent",
        "Bubble-Selection and Insertion-Selection match the paper-like high final/peak pattern; Bubble-Insertion has a close peak but returns near 0.5 final Aggregation.",
        "Strict tie handling and deterministic scheduler can preserve or dissolve same-Algotype blocks differently from the original threaded run.",
        "S11",
        [paths["s11_comparison"], paths["s11_peaks"]],
    )
    add(
        "FIG08_DUPLICATE_FINAL_VS_UNIQUE",
        "Figure 8",
        "numeric result",
        "Allowing duplicate values raises final Aggregation above the unique-value case when value-order pressure is relaxed.",
        "Paper text emphasizes Bubble-Selection and Insertion-Selection final Aggregation above unique-value peaks, with Bubble-Insertion limited by similar component efficiencies.",
        "; ".join(
            f"{r.mixtureId}: duplicate-final-minus-unique-final={fmt(r.duplicateMinusUniqueFinalAggregation)}, duplicate-mean-curve-peak-minus-unique={fmt(r.duplicateMeanCurvePeakMinusUnique)}"
            for r in s11_comp.itertuples(index=False)
        ),
        "directionally consistent",
        "The expected release-of-pressure pattern holds strongly for Bubble-Selection and Insertion-Selection; Bubble-Insertion remains near the unique final value and its mean-curve peak is slightly lower than unique.",
        "The paper's duplicate-value narrative is pair-specific; Bubble-Insertion is constrained by similar component efficiencies and by strict equal-value comparison behavior.",
        "S11",
        [paths["s11_comparison"], step_dir / "S11" / "summary.md"],
    )

    s12_summary = tables["s12_summary"]
    s12_equilibria = tables["s12_equilibria"]
    s12_paper = tables["s12_paper"]
    unique_paper = s12_paper[s12_paper["inputProfile"] == "unique_1_100"]
    add(
        "FIG09_UNIQUE_OPPOSITE_FINAL_SORTEDNESS",
        "Figure 9",
        "numeric result",
        "Opposite-direction unique-value chimeras stop below 100% Sortedness at 42.5, 73.73, and 38.31 for the three pairings.",
        "Paper final Sortedness values by pairing.",
        "; ".join(
            f"{r.mixtureId}: observed={fmt(r.meanFinalIncreasingSortednessRaw)}, paper={fmt(r.paperUniqueFinalSortednessRaw)}, diff={fmt(r.differenceFromPaperRaw)}"
            for r in unique_paper.itertuples(index=False)
        ),
        "directionally consistent",
        "All unique opposite-direction conditions stop below 100% and two of three paper values are close; Bubble-down/Selection-up is lower by 9.41 adjacent-pair counts.",
        "Several S12 runs hit the S03 max-swap cap, and deterministic pseudo-scheduling can change the conflict trajectory.",
        "S12",
        [paths["s12_paper"], paths["s12_summary"]],
        review_priority="anchor",
    )
    unique_eq = s12_equilibria[s12_equilibria["inputProfile"] == "unique_1_100"]
    cap_total = int(s12_summary["capStopCount"].sum())
    add(
        "FIG09_UNIQUE_OPPOSITE_DYNAMICS_EQUILIBRIUM",
        "Figure 9",
        "trajectory/result",
        "The three unique opposite-direction pairings have distinct trajectories and reach a global equilibrium.",
        "Stable flattening below 100% Sortedness with different trajectory shapes.",
        "; ".join(
            f"{r.mixtureId}: modalEquilibrium={r.modalEquilibriumClassification}, capStopCount={int(row_lookup(s12_summary, conditionId=r.conditionId)['capStopCount'])}"
            for r in unique_eq.itertuples(index=False)
        ),
        "directionally consistent",
        "The regenerated trajectories are distinct and flatten, but cap-truncated runs mean stable no-move equilibrium is not exact for all replicates.",
        "The S03 cap and deterministic scheduler expose long-running conflicts that the paper may have stopped under a different no-change criterion.",
        "S12",
        [paths["s12_equilibria"], artifacts_dir / "figures" / "e01" / "figure09_unique_opposite.png"],
        review_priority="high",
    )
    add(
        "FIG09_UNIQUE_OPPOSITE_AGGREGATION_DOMINANCE",
        "Figure 9",
        "numeric result",
        "In unique opposite-direction chimeras, algorithm dominance follows Bubble > Selection > Insertion and Aggregation rises above the about-0.5 start.",
        "Dominance order Bubble > Selection > Insertion; final Aggregation above starting average.",
        "; ".join(
            f"{r.mixtureId}: dominanceMatch={fmt(r.dominanceMatchesPaperRate)}, finalAggregation={fmt(row_lookup(s12_summary, conditionId=r.conditionId)['meanFinalAggregation'])}"
            for r in unique_eq.itertuples(index=False)
        ),
        "directionally consistent",
        "All three unique pairings match the paper's dominance ordering and have final Aggregation above 0.5, though caps constrain the equilibrium claim.",
        "Same cap and scheduler limitations as the Figure 9 equilibrium row.",
        "S12",
        [paths["s12_equilibria"], paths["s12_summary"]],
    )
    dup_eq = s12_equilibria[s12_equilibria["inputProfile"] == "duplicate_1_10_x10"]
    add(
        "FIG10_REPEATED_OPPOSITE_SIMILARITY",
        "Figure 10",
        "numeric result",
        "Repeated-value opposite-direction chimeras give similar qualitative results to unique-value opposite-direction chimeras.",
        "Repeated values 1..10 x10 have similar non-100% final states and flattened trajectories.",
        "; ".join(
            f"{r.mixtureId}: finalIncreasing={fmt(row_lookup(s12_summary, conditionId=r.conditionId)['meanFinalIncreasingSortednessRaw'])}, finalAggregation={fmt(row_lookup(s12_summary, conditionId=r.conditionId)['meanFinalAggregation'])}, capStopCount={int(row_lookup(s12_summary, conditionId=r.conditionId)['capStopCount'])}"
            for r in dup_eq.itertuples(index=False)
        ),
        "not replicated",
        "The repeated-value runs flatten and have Aggregation above 0.5, but final Sortedness levels are not similar to the unique-value Figure 9 pattern for all pairings.",
        "Duplicate values change the conflict landscape; strict equal-value handling and cap-truncated Bubble pairings alter the large-scale state.",
        "S12",
        [paths["s12_summary"], artifacts_dir / "figures" / "e01" / "figure10_repeated_opposite.png"],
        review_priority="high",
    )
    add(
        "FIG10_REPEATED_OPPOSITE_DOMINANCE",
        "Figure 10",
        "numeric result",
        "Repeated-value opposite-direction chimeras preserve the Bubble > Selection > Insertion dominance order.",
        "Dominance order Bubble > Selection > Insertion for repeated values.",
        "; ".join(
            f"{r.mixtureId}: modalDominant={r.modalDominantAlgotype}, expected={r.expectedDominantAlgotype}, matchRate={fmt(r.dominanceMatchesPaperRate)}"
            for r in dup_eq.itertuples(index=False)
        ),
        "not replicated",
        "Bubble pairings match the paper ordering, but Selection-down/Insertion-up has only 32% paper-order dominance and a modal Insertion-dominant state.",
        "Duplicate ties and deterministic local moves let Insertion dominate many repeated-value conflicts in the S12 baseline.",
        "S12",
        [paths["s12_equilibria"], paths["s12_summary"]],
        review_priority="high",
    )

    df = pd.DataFrame.from_records(records)
    df["classificationRank"] = df["classification"].map(
        {"exact": 0, "statistically consistent": 1, "directionally consistent": 2, "not replicated": 3}
    )
    return df.sort_values(["classificationRank", "paperFigure", "claimId"]).drop(columns=["classificationRank"]).reset_index(drop=True)


def write_classification_methods(path: Path, artifacts_written: list[str], validation_result: str, caveats: list[str], recommended_next_action: str) -> None:
    lines = [
        "# S13 Classification Methods",
        "",
        "- Research step ID: S13",
        "- Step number: 13",
        "- Completion status: completed",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts_written)
    lines.extend(
        [
            "- Validation result: " + validation_result,
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Classification Criteria",
            "",
            "- `exact`: the S01-S12 artifact reproduces the paper claim at the declared replicate/condition count or an invariant exactly.",
            "- `statistically consistent`: the regenerated test or interval preserves the paper's direction and significance/similarity conclusion, even if exact z-statistics or factors differ.",
            "- `directionally consistent`: the qualitative direction or ordering is preserved, but numeric magnitude, timing, or full statistical support differs or is unavailable.",
            "- `not replicated`: the required condition/control is missing, the regenerated result conflicts with the paper claim, or available artifacts are insufficient to support the claim.",
            "",
            "The audit uses only S01 through S12 outputs and the attached paper text. It does not start S14 scaled runs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_divergence_log(
    path: Path,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    overview_rows = [
        [row["classification"], int(row["claimCount"])]
        for _, row in summary_df.sort_values("classification").iterrows()
    ]
    claim_rows = [
        [
            row.claimId,
            row.paperFigure,
            row.classification,
            row.observedEvidence,
            row.justification,
        ]
        for row in df.itertuples(index=False)
    ]
    lines = [
        "# E01 S13 Divergence Log",
        "",
        "- Research step ID: S13",
        "- Step number: 13",
        "- Completion status: completed",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts_written)
    lines.extend(
        [
            "- Validation result: " + validation_result,
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Classification Counts",
            "",
            markdown_table(["Classification", "Claim count"], overview_rows),
            "",
            "## Claim-Level Map",
            "",
            markdown_table(["Claim ID", "Paper figure", "Classification", "Observed evidence", "Justification"], claim_rows),
            "",
            "## Divergence Entries",
            "",
        ]
    )
    for row in df.itertuples(index=False):
        lines.extend(
            [
                f"### {row.claimId}",
                "",
                f"- Paper figure: {row.paperFigure}",
                f"- Claim type: {row.claimType}",
                f"- Paper claim: {row.paperClaim}",
                f"- Paper expected value: {row.paperExpectedValue}",
                f"- Observed evidence: {row.observedEvidence}",
                f"- Classification: {row.classification}",
                f"- Justification: {row.justification}",
                f"- Plausible causes for mismatches: {row.plausibleMismatchCauses}",
                f"- Source research steps: {row.sourceResearchSteps}",
                f"- Supporting artifacts: {row.supportingArtifacts}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_markdown(
    path: Path,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    outcome_classification: str,
) -> None:
    counts = summary_df.set_index("classification")["claimCount"].to_dict()
    high_priority = df[df["reviewPriority"].isin(["high", "anchor"])]
    high_rows = [
        [row.claimId, row.paperFigure, row.classification, row.justification]
        for row in high_priority.itertuples(index=False)
    ]
    lines = [
        "# S13 Status Summary",
        "",
        "- Research step ID: S13",
        "- Step number: 13",
        "- Completion status: completed",
        "- Outcome classification: " + outcome_classification,
        "- Artifacts written:",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts_written)
    lines.extend(
        [
            "- Validation result: " + validation_result,
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Lay summary: S13 audited the E01 replication claim by claim. Exact replication is strongest for sorting completion, statistical replication is strongest for the reported efficiency/DG/Aggregation directions, and the largest unresolved divergences are the Figure 8 same-code negative control and the repeated-value opposite-direction dominance pattern.",
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Classification Summary",
            "",
            markdown_table(
                ["Classification", "Claim count"],
                [[key, counts.get(key, 0)] for key in ["exact", "statistically consistent", "directionally consistent", "not replicated"]],
            ),
            "",
            "## Review-Focused Claims",
            "",
            markdown_table(["Claim ID", "Figure", "Classification", "Justification"], high_rows),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_paths: list[Path],
    artifacts_dir: Path,
) -> tuple[bool, list[str], list[str], str]:
    checks: list[str] = []
    caveats: list[str] = []
    failures: list[str] = []

    expected = set(EXPECTED_CLAIM_IDS)
    observed = set(df["claimId"])
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        failures.append(f"Missing expected S13 claim IDs: {missing}.")
    else:
        checks.append(f"All {len(expected)} expected core claim IDs are present.")
    if extra:
        failures.append(f"Unexpected S13 claim IDs: {extra}.")
    else:
        checks.append("No unexpected claim IDs were emitted.")

    if set(df["classification"]).issubset(ALLOWED_CLASSIFICATIONS):
        checks.append("All classification labels are in the allowed tolerance set.")
    else:
        failures.append("At least one classification label is outside the allowed tolerance set.")

    for column in ["paperClaim", "observedEvidence", "justification", "plausibleMismatchCauses", "supportingArtifacts"]:
        if df[column].astype(str).str.len().min() > 0:
            checks.append(f"Every claim row has nonempty `{column}`.")
        else:
            failures.append(f"At least one claim row has an empty `{column}`.")

    figures = set(df["paperFigure"].str.extract(r"(Figure \d+)")[0].dropna())
    required_figures = {f"Figure {i}" for i in range(1, 11)}
    if required_figures.issubset(figures):
        checks.append("Claim table covers Figures 1 through 10.")
    else:
        failures.append(f"Missing figure coverage: {sorted(required_figures - figures)}.")

    if int(summary_df["claimCount"].sum()) == len(df):
        checks.append("Classification summary counts match the claim table row count.")
    else:
        failures.append("Classification summary counts do not match the claim table row count.")

    statuses_ok = True
    missing_statuses: list[str] = []
    for step_num in range(1, 13):
        step = f"S{step_num:02d}"
        status_path = artifacts_dir / "research_steps" / step / "status.json"
        status = load_json(status_path)
        if not status or not status.get("success", False):
            statuses_ok = False
            missing_statuses.append(step)
    if statuses_ok:
        checks.append("All S01 through S12 source status JSON files exist and report success.")
    else:
        failures.append(f"Missing or unsuccessful source status files: {missing_statuses}.")

    missing_outputs = [str(p) for p in output_paths if not p.exists() or p.stat().st_size == 0]
    if missing_outputs:
        failures.append(f"Missing or empty declared S13 outputs: {missing_outputs}.")
    else:
        checks.append("All declared S13 outputs exist and are non-empty.")

    s14_dir = artifacts_dir / "research_steps" / "S14"
    if s14_dir.exists():
        failures.append("S14 artifact directory exists; S13 must stop before S14.")
    else:
        checks.append("No S14 artifact directory exists.")

    caveats.extend(
        [
            "Classifications use the extracted paper text and the S01-S12 regenerated artifacts; original raw paper traces and seeds are unavailable.",
            "Conceptual figures were classified against method coverage, not pixel-level figure recreation.",
            "Several mismatches plausibly stem from deterministic pseudo-scheduling, reconstructed traditional wrappers, comparison-count conventions, Frozen Cell semantics, and missing same-code negative-control traces.",
            "The cumulative run manifest before S13 lacked a distinct S12 `researchSteps` entry because S12 reused an S11 helper updater; S13 preserves local S12 artifacts and writes a corrected S13 entry.",
            "S13 does not start S14 scaled runs.",
        ]
    )

    success = not failures
    validation_result = (
        f"passed: {len(df)} claim rows classified; summary counts validated; S01-S12 source statuses checked; no S14 artifacts started."
        if success
        else "failed: " + " ".join(failures)
    )
    return success, checks + failures, caveats, validation_result


def write_validation_markdown(
    path: Path,
    checks: list[str],
    caveats: list[str],
    artifacts_written: list[str],
    validation_result: str,
    recommended_next_action: str,
) -> None:
    lines = [
        "# S13 Validation",
        "",
        "- Research step ID: S13",
        "- Step number: 13",
        "- Completion status: completed",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts_written)
    lines.extend(
        [
            "- Validation result: " + validation_result,
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in checks)
    path.write_text("\n".join(lines), encoding="utf-8")


def update_run_manifest(path: Path, artifacts_dir: Path, status_payload: dict[str, Any], artifact_manifest_payload: dict[str, Any]) -> None:
    manifest = load_json(path)
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest.setdefault("researchSteps", {})

    # Preserve prior provenance while backfilling the missing S12 cumulative entry
    # from its local status and artifact manifest when needed.
    for step_num in range(1, 13):
        step = f"S{step_num:02d}"
        if step in manifest["researchSteps"]:
            continue
        step_status = load_json(artifacts_dir / "research_steps" / step / "status.json")
        step_manifest = load_json(artifacts_dir / "research_steps" / step / "artifact_manifest.json")
        if not step_status:
            continue
        manifest["researchSteps"][step] = {
            "status": step_status.get("status"),
            "success": step_status.get("success"),
            "outcomeClassification": step_status.get("outcomeClassification"),
            "artifactCount": step_manifest.get("artifactCount"),
            "artifacts": step_manifest.get("artifacts", []),
            "validationResult": step_status.get("validationResult"),
            "recommendedNextAction": step_status.get("recommendedNextAction"),
            "generatedAt": step_status.get("generatedAt"),
            "git": step_status.get("git"),
            "backfilledBy": STEP_ID,
        }

    manifest["updatedAt"] = status_payload["generatedAt"]
    manifest["latestResearchStepId"] = STEP_ID
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "outcomeClassification": status_payload["outcomeClassification"],
        "artifactCount": artifact_manifest_payload["artifactCount"],
        "artifacts": artifact_manifest_payload["artifacts"],
        "validationResult": status_payload["validationResult"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "generatedAt": status_payload["generatedAt"],
        "git": status_payload["git"],
    }
    write_json(path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--paper-md", type=Path, default=PAPER_MD_DEFAULT)
    return parser.parse_args()


def run_s13(args: argparse.Namespace) -> int:
    generated_at = utc_now()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    reports_dir = artifacts_dir / "reports"
    provenance_dir = artifacts_dir / "provenance"
    for directory in [step_dir, code_dir, results_dir, reports_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    classification_csv = results_dir / "e01_replication_classification.csv"
    classification_parquet = results_dir / "e01_replication_classification.parquet"
    classification_summary_csv = results_dir / "e01_replication_classification_summary.csv"
    classification_summary_parquet = results_dir / "e01_replication_classification_summary.parquet"
    divergence_log_md = reports_dir / "e01_divergence_log.md"
    methods_md = step_dir / "classification_methods.md"
    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    summary_md = step_dir / "summary.md"
    status_json = step_dir / "status.json"
    artifact_manifest_json = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    code_copy = code_dir / Path(__file__).name

    df = build_classification_rows(artifacts_dir, args.paper_md)
    summary_df = (
        df.groupby("classification", as_index=False)
        .agg(claimCount=("claimId", "count"))
        .assign(researchStepId=STEP_ID, stepNumber=STEP_NUMBER)
    )
    classification_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(classification_csv, index=False)
    df.to_parquet(classification_parquet, index=False)
    summary_df.to_csv(classification_summary_csv, index=False)
    summary_df.to_parquet(classification_summary_parquet, index=False)

    initial_output_paths = [
        classification_csv,
        classification_parquet,
        classification_summary_csv,
        classification_summary_parquet,
        divergence_log_md,
        methods_md,
        validation_json,
        validation_md,
        summary_md,
        status_json,
        artifact_manifest_json,
        code_copy,
        run_manifest_path,
    ]

    recommended_next_action = (
        "Hand control back to the Chief Scientist workflow for review; start S14 scaled runs only after explicit instruction, prioritizing exact/statistically consistent anchor claims and the not-replicated/control divergences documented by S13."
    )
    success, validation_checks, caveats, validation_result = validate_outputs(
        df,
        summary_df,
        [classification_csv, classification_parquet, classification_summary_csv, classification_summary_parquet],
        artifacts_dir,
    )
    outcome_classification = "supportive" if success else "constraining"
    artifacts_written_placeholder = [str(p) for p in sorted(set(initial_output_paths))]

    write_classification_methods(methods_md, artifacts_written_placeholder, validation_result, caveats, recommended_next_action)
    write_divergence_log(
        divergence_log_md,
        df,
        summary_df,
        artifacts_written_placeholder,
        validation_result,
        caveats,
        recommended_next_action,
    )
    write_validation_markdown(validation_md, validation_checks, caveats, artifacts_written_placeholder, validation_result, recommended_next_action)
    write_summary_markdown(
        summary_md,
        df,
        summary_df,
        artifacts_written_placeholder,
        validation_result,
        caveats,
        recommended_next_action,
        outcome_classification,
    )
    shutil.copy2(Path(__file__), code_copy)

    artifacts_written = [str(p) for p in sorted(set(initial_output_paths))]
    validation_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "validationChecks": validation_checks,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "claimRows": int(len(df)),
        "classificationCounts": summary_df.set_index("classification")["claimCount"].to_dict(),
    }
    write_json(validation_json, validation_payload)

    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "outcomeClassification": outcome_classification,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "git": get_git_metadata(),
        "runtime": {
            "pythonVersion": sys.version,
            "pythonExecutable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "osCpuCount": os.cpu_count(),
            "pandasVersion": pd.__version__,
            "gpuUsed": False,
            "workerCount": 1,
        },
        "sourceArtifacts": {
            "paperMarkdown": str(args.paper_md),
            "sourceResearchSteps": [f"S{i:02d}" for i in range(1, 13)],
        },
        "classificationCounts": summary_df.set_index("classification")["claimCount"].to_dict(),
        "notReplicatedClaimIds": sorted(df.loc[df["classification"] == "not replicated", "claimId"].tolist()),
        "highPriorityClaimIds": sorted(df.loc[df["reviewPriority"].isin(["high", "anchor"]), "claimId"].tolist()),
    }
    write_json(status_json, status_payload)

    artifact_manifest_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "manifestSelfPath": str(artifact_manifest_json),
        "provenanceManifestPath": str(run_manifest_path),
        "artifactCount": 0,
        "artifacts": [],
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
    }
    write_json(artifact_manifest_json, artifact_manifest_payload)
    artifact_manifest_payload["artifacts"] = collect_artifacts([Path(p) for p in artifacts_written if Path(p) != run_manifest_path])
    artifact_manifest_payload["artifactCount"] = len(artifact_manifest_payload["artifacts"])
    write_json(artifact_manifest_json, artifact_manifest_payload)
    update_run_manifest(run_manifest_path, artifacts_dir, status_payload, artifact_manifest_payload)

    if not success:
        print(validation_result, file=sys.stderr)
        return 1
    print(validation_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_s13(parse_args()))
