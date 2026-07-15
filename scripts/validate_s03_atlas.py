#!/usr/bin/env python3
"""Independently validate the compact E03 S03 metric-disagreement evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


S02 = Path("/artifacts/research_steps/S02")
METRICS = [
    "adjacent_descents",
    "normalized_adjacent_descents",
    "paper_sortedness_distance",
    "inversion_count",
    "normalized_kendall_distance",
    "spearman_footrule",
    "normalized_spearman_footrule",
    "maximum_rank_error",
    "normalized_maximum_rank_error",
    "duplicate_aware_earth_movers_distance",
    "normalized_duplicate_aware_earth_movers_distance",
]
SIGN_PAIRS = [
    ("adjacent_descents", "normalized_adjacent_descents"),
    ("inversion_count", "normalized_kendall_distance"),
    ("spearman_footrule", "normalized_spearman_footrule"),
    ("maximum_rank_error", "normalized_maximum_rank_error"),
    ("spearman_footrule", "duplicate_aware_earth_movers_distance"),
    (
        "normalized_spearman_footrule",
        "normalized_duplicate_aware_earth_movers_distance",
    ),
]
BOUNDARY_RULES = {
    "plateau_bridge_initial_included",
    "plateau_break_initial_included",
    "plateau_bridge_post_swap_only",
    "plateau_break_post_swap_only",
}
TOLERANCE = 1e-12
BOOTSTRAP_SEED = 3_761_897_473
BOOTSTRAP_DRAWS = 10_000


def sign(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return np.where(
        array > TOLERANCE,
        "worsening",
        np.where(array < -TOLERANCE, "improvement", "neutral"),
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def independent_bootstrap(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    label: str,
) -> tuple[float, float, float]:
    """Recreate one clustered ratio interval without importing atlas helpers."""

    contributions = (
        frame.groupby("logical_trace_id", sort=True)[[numerator, denominator]]
        .sum()
        .to_numpy(dtype=float)
    )
    estimate = float(contributions[:, 0].sum() / contributions[:, 1].sum())
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    seed = (BOOTSTRAP_SEED ^ int.from_bytes(digest[:4], "big")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 1000):
        stop = min(start + 1000, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, len(contributions), size=(stop - start, len(contributions)))
        selected = contributions[indices]
        numerator_sum = selected[:, :, 0].sum(axis=1)
        denominator_sum = selected[:, :, 1].sum(axis=1)
        values[start:stop] = np.divide(
            numerator_sum,
            denominator_sum,
            out=np.full(stop - start, np.nan),
            where=denominator_sum > 0,
        )
    lower, upper = np.quantile(values[np.isfinite(values)], [0.025, 0.975])
    return estimate, float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/artifacts/research_steps/S03"),
    )
    args = parser.parse_args()
    root = args.artifact_dir
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"checkId": check_id, "passed": bool(passed), "detail": detail})

    required = [
        "metric_disagreement_atlas.parquet",
        "metric_disagreement_episodes.parquet",
        "metric_sign_summary.parquet",
        "episode_metric_sign_summary.parquet",
        "bootstrap_uncertainty.parquet",
        "representative_traces.parquet",
        "episode_boundary_sensitivity.csv",
        "denominator_sensitivity.csv",
        "metric_dependence.csv",
        "mixed_direction_projection_sensitivity.csv",
        "coverage_gap_sensitivity.json",
        "representative_trace_manifest.json",
        "event_sign_heatmap.png",
        "event_sign_heatmap.svg",
        "episode_sign_heatmap.png",
        "episode_sign_heatmap.svg",
        "representative_traces.png",
        "representative_traces.svg",
        "analysis_summary.json",
        "input_immutability.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    empty = [name for name in required if (root / name).is_file() and not (root / name).stat().st_size]
    check("required_outputs_present", not missing and not empty, f"missing={missing}; empty={empty}")

    events = pq.read_table(root / "metric_disagreement_atlas.parquet").to_pandas()
    episodes = pq.read_table(root / "metric_disagreement_episodes.parquet").to_pandas()
    event_signs = pq.read_table(root / "metric_sign_summary.parquet").to_pandas()
    bootstrap = pq.read_table(root / "bootstrap_uncertainty.parquet").to_pandas()
    episode_signs = pq.read_table(root / "episode_metric_sign_summary.parquet").to_pandas()
    representatives = pq.read_table(root / "representative_traces.parquet").to_pandas()
    source = pq.read_table(
        S02 / "replayed_distances.parquet",
        columns=[
            "checkpoint_kind",
            "logical_trace_id",
            "direction",
            "accepted_swap_index",
        ],
    ).to_pandas()
    source_actions = source[source["checkpoint_kind"] == "accepted_action"]
    check(
        "event_row_bijection",
        len(events) == len(source_actions),
        f"atlas={len(events)}; S02 accepted-action projections={len(source_actions)}",
    )
    check(
        "aligned_trace_coverage",
        events["logical_trace_id"].nunique() == 85,
        f"logical traces={events['logical_trace_id'].nunique()}; expected=85",
    )
    primary_event_signs = event_signs[event_signs["population"] == "primary_homogeneous"]
    expected_event_strata = {
        "overall",
        "source_step_id",
        "architecture",
        "policy_composition",
        "condition_family",
        "fault_mode",
        "realized_fault_count",
        "scheduler",
        "input_profile",
        "stop_reason",
        "initial_kendall_disorder_bin",
        "action_progress_tertile",
        "goal_direction_scope",
        "direction",
    }
    check(
        "event_all_metrics_and_strata",
        set(primary_event_signs["metric"].unique()) == set(METRICS)
        and set(primary_event_signs["stratum"].unique()) == expected_event_strata,
        (
            f"metrics={primary_event_signs['metric'].nunique()}; "
            f"strata={primary_event_signs['stratum'].nunique()}"
        ),
    )
    source_keys = set(
        map(tuple, source_actions[["logical_trace_id", "direction", "accepted_swap_index"]].to_numpy())
    )
    event_keys = set(
        map(tuple, events[["logical_trace_id", "direction", "accepted_swap_index"]].to_numpy())
    )
    check(
        "event_key_identity",
        source_keys == event_keys and len(event_keys) == len(events),
        f"source keys={len(source_keys)}; atlas unique keys={len(event_keys)}",
    )

    continuity_errors = 0
    for _, group in events.groupby(["logical_trace_id", "direction"], sort=False):
        ordered = group.sort_values("accepted_swap_index")
        expected = np.arange(1, len(ordered) + 1)
        continuity_errors += int(np.count_nonzero(ordered["accepted_swap_index"].to_numpy() != expected))
        continuity_errors += int(
            np.count_nonzero(
                ordered["previous_accepted_swap_index"].to_numpy()
                != np.arange(0, len(ordered))
            )
        )
    check(
        "action_predecessor_continuity",
        continuity_errors == 0,
        f"continuity mismatches={continuity_errors}",
    )

    sign_errors = 0
    for metric in METRICS:
        sign_errors += int(
            np.count_nonzero(sign(events[f"delta_{metric}"]) != events[f"sign_{metric}"].to_numpy())
        )
    check("delta_sign_recalculation", sign_errors == 0, f"mismatches={sign_errors}")
    dependency_errors = {
        f"{left}__{right}": int(
            np.count_nonzero(events[f"sign_{left}"] != events[f"sign_{right}"])
        )
        for left, right in SIGN_PAIRS
    }
    check(
        "correlated_metric_sign_identity",
        all(value == 0 for value in dependency_errors.values()),
        json.dumps(dependency_errors, sort_keys=True),
    )

    expected_paper = events["sign_paper_sortedness_distance"] == "worsening"
    check(
        "paper_backtrack_fixture_on_atlas",
        bool((expected_paper == events["paper_backtrack"]).all()),
        f"paper backtracks={int(events['paper_backtrack'].sum())}",
    )
    expected_global = pd.concat(
        [events[f"sign_{metric}"] == "worsening" for metric in (
            "inversion_count", "spearman_footrule", "maximum_rank_error"
        )],
        axis=1,
    ).any(axis=1)
    check(
        "independent_global_consensus_recalculation",
        bool((expected_global == events["any_independent_global_regression"]).all()),
        f"global-regression actions={int(expected_global.sum())}",
    )

    check(
        "episode_rule_coverage",
        set(episodes["boundary_rule"].unique()) == BOUNDARY_RULES,
        f"rules={sorted(episodes['boundary_rule'].unique())}",
    )
    primary_episode_signs = episode_signs[
        episode_signs["population"] == "primary_homogeneous"
    ]
    expected_episode_strata = {
        "overall",
        "source_step_id",
        "architecture",
        "policy_composition",
        "condition_family",
        "fault_mode",
        "realized_fault_count",
        "scheduler",
        "input_profile",
        "stop_reason",
        "initial_kendall_disorder_bin",
        "episode_position_tertile",
        "goal_direction_scope",
        "direction",
    }
    check(
        "episode_all_metrics_and_strata",
        set(primary_episode_signs["metric"].unique()) == set(METRICS)
        and set(primary_episode_signs["boundary_rule"].unique()) == BOUNDARY_RULES
        and set(primary_episode_signs["stratum"].unique()) == expected_episode_strata
        and set(primary_episode_signs["phase"].unique()) == {"excursion", "end"},
        (
            f"metrics={primary_episode_signs['metric'].nunique()}; "
            f"rules={primary_episode_signs['boundary_rule'].nunique()}; "
            f"strata={primary_episode_signs['stratum'].nunique()}"
        ),
    )
    episode_valid = (
        (episodes["start_action_index"] <= episodes["worsening_end_action_index"])
        & (episodes["worsening_end_action_index"] <= episodes["end_action_index"])
        & (episodes["paper_worsening_magnitude"] > 0)
        & (episodes["paper_recovery_magnitude"] >= 0)
    )
    check(
        "episode_boundaries_and_magnitudes",
        bool(episode_valid.all()) and episodes["episode_id"].is_unique,
        f"invalid={int((~episode_valid).sum())}; duplicate IDs={int(episodes['episode_id'].duplicated().sum())}",
    )

    finite_bootstrap = bootstrap.dropna(subset=["estimate"])
    interval_valid = (
        finite_bootstrap["estimate"].between(0, 1)
        & finite_bootstrap["ci_lower"].between(0, 1)
        & finite_bootstrap["ci_upper"].between(0, 1)
        & (finite_bootstrap["ci_lower"] <= finite_bootstrap["ci_upper"])
    )
    check(
        "bootstrap_interval_bounds",
        bool(interval_valid.all()),
        f"estimates={len(finite_bootstrap)}; invalid={int((~interval_valid).sum())}",
    )
    check(
        "bootstrap_draw_contract",
        set(bootstrap["bootstrap_draws"].unique()) == {10_000}
        and set(bootstrap["bootstrap_unit"].unique()) == {"logical_trace"},
        f"draws={sorted(bootstrap['bootstrap_draws'].unique())}; units={sorted(bootstrap['bootstrap_unit'].unique())}",
    )
    overall = bootstrap[(bootstrap["stratum"] == "overall") & (bootstrap["level"] == "all")]
    check(
        "bootstrap_overall_metric_coverage",
        len(overall) == 8,
        f"overall estimates={len(overall)}; expected=8",
    )
    primary = events[
        events["is_primary_release_trace"] & events["is_homogeneous_native_goal"]
    ].copy()
    primary["denom_action"] = 1
    primary["num_paper_backtrack"] = primary["paper_backtrack"].astype(int)
    primary["denom_paper_backtrack"] = primary["paper_backtrack"].astype(int)
    primary["num_local_only"] = (
        primary["event_consensus_class"] == "paper_backtrack_all_global_nonworsening"
    ).astype(int)
    primary["num_global_regression"] = primary["any_independent_global_regression"].astype(int)
    primary["num_adjacent_disagreement"] = primary["paper_adjacent_sign_disagreement"].astype(int)
    event_specs = {
        "paper_backtrack_rate": ("num_paper_backtrack", "denom_action"),
        "paper_backtrack_all_global_nonworsening_rate": ("num_local_only", "denom_paper_backtrack"),
        "any_global_regression_rate": ("num_global_regression", "denom_action"),
        "paper_adjacent_sign_disagreement_rate": ("num_adjacent_disagreement", "denom_action"),
    }
    primary_episodes = episodes[
        (episodes["boundary_rule"] == "plateau_bridge_initial_included")
        & episodes["is_primary_release_trace"]
        & episodes["is_homogeneous_native_goal"]
    ].copy()
    primary_episodes["denom_episode"] = 1
    primary_episodes["num_global_nonworsening"] = primary_episodes[
        "all_independent_globals_nonworsening"
    ].astype(int)
    primary_episodes["num_global_excursion"] = primary_episodes[
        "any_independent_global_excursion"
    ].astype(int)
    primary_episodes["num_paired"] = primary_episodes["paired_recovery"].astype(int)
    primary_episodes["denom_paired"] = primary_episodes["paired_recovery"].astype(int)
    primary_episodes["num_net_positive"] = (
        primary_episodes["paired_recovery"]
        & (primary_episodes["paper_net_recovery_score"] > TOLERANCE)
    ).astype(int)
    episode_specs = {
        "paper_episode_all_global_nonworsening_rate": ("num_global_nonworsening", "denom_episode"),
        "paper_episode_any_global_excursion_rate": ("num_global_excursion", "denom_episode"),
        "paired_recovery_rate": ("num_paired", "denom_episode"),
        "positive_net_recovery_rate_among_paired": ("num_net_positive", "denom_paired"),
    }
    bootstrap_mismatches: list[str] = []
    for analysis_unit, frame, specs, label_prefix in (
        ("accepted_action", primary, event_specs, "event"),
        ("paper_episode", primary_episodes, episode_specs, "episode"),
    ):
        for metric, (numerator, denominator) in specs.items():
            expected = independent_bootstrap(
                frame,
                numerator,
                denominator,
                f"{label_prefix}|overall|all|{metric}",
            )
            row = overall[
                (overall["analysis_unit"] == analysis_unit) & (overall["metric"] == metric)
            ].iloc[0]
            observed = (row["estimate"], row["ci_lower"], row["ci_upper"])
            if not np.allclose(expected, observed, rtol=0, atol=1e-15):
                bootstrap_mismatches.append(f"{metric}: expected={expected}, observed={observed}")
    check(
        "bootstrap_independent_overall_reconstruction",
        not bootstrap_mismatches,
        "all 8 overall 10,000-draw intervals reproduced"
        if not bootstrap_mismatches
        else "; ".join(bootstrap_mismatches),
    )

    gaps = json.loads((root / "coverage_gap_sensitivity.json").read_text())
    check(
        "six_coverage_gaps_explicit",
        gaps["gapTraceCount"] == 6
        and len(gaps["traces"]) == 6
        and all(row["globalClassification"] == "unavailable_not_imputed" for row in gaps["traces"]),
        f"gap traces={gaps['gapTraceCount']}; global classifications are not imputed",
    )
    denominator = pd.read_csv(root / "denominator_sensitivity.csv")
    gap_bound = denominator[
        denominator["analysis"] == "local_only_fraction_among_paper_backtracks_gap_bounds"
    ].iloc[0]
    check(
        "coverage_gap_bounds",
        0 <= gap_bound["lower_bound"] <= gap_bound["upper_bound"] <= 1,
        f"bounds=[{gap_bound['lower_bound']}, {gap_bound['upper_bound']}]",
    )

    mixed = events[events["goal_direction_scope"] != "native_homogeneous_goal"]
    pair_counts = mixed.groupby(["logical_trace_id", "accepted_swap_index"])["direction"].agg(
        lambda values: tuple(sorted(values))
    )
    check(
        "mixed_direction_pairing",
        mixed["logical_trace_id"].nunique() == 12
        and len(pair_counts)
        and all(value == ("ascending", "descending") for value in pair_counts),
        f"mixed logical traces={mixed['logical_trace_id'].nunique()}; paired actions={len(pair_counts)}",
    )

    immutable = json.loads((root / "input_immutability.json").read_text())
    check(
        "input_immutability",
        immutable["success"] is True
        and immutable["preRunSha256"] == immutable["postRunSha256"],
        f"success={immutable['success']}",
    )
    rep_manifest = json.loads((root / "representative_trace_manifest.json").read_text())
    selected = [row for row in rep_manifest["selections"] if row["status"] == "selected"]
    check(
        "representative_windows",
        bool(selected) and not representatives.empty,
        f"selected anchors={len(selected)}; window rows={len(representatives)}",
    )

    passed = all(row["passed"] for row in checks)
    result = {
        "schemaVersion": "e03.s03.validation_results.v1",
        "researchStepId": "S03",
        "success": passed,
        "validationResult": f"{sum(row['passed'] for row in checks)}/{len(checks)} checks passed",
        "checks": checks,
    }
    write_json(root / "validation_results.json", result)
    lines = [f"[{('PASS' if row['passed'] else 'FAIL')}] {row['checkId']}: {row['detail']}" for row in checks]
    (root / "validation.log").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
