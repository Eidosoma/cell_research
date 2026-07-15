#!/usr/bin/env python3
"""Build the E03 S03 event/episode metric-disagreement atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from src.detours.atlas import (
    IMPROVEMENT,
    NEUTRAL,
    WORSENING,
    clustered_percentile_interval,
    delta_sign,
    paper_episode_boundaries,
    stable_seed,
)


S02 = Path("/artifacts/research_steps/S02")
E01 = Path("/previous-artifacts/E01")
SCENARIO_BANK = E01 / "scenarios/paired_scenario_bank.parquet"
GAP_TRACES = E01 / "research_steps/S09/selected_traces.jsonl.zst"
CONTRACT = Path(__file__).resolve().parents[1] / "analysis/s03_atlas_contract.json"
SCHEMA_VERSION = "e03.s03.metric_disagreement_event.v1"
EPISODE_SCHEMA = "e03.s03.metric_disagreement_episode.v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 3_761_897_473
TOLERANCE = 1e-12

DISTANCE_VARIANTS = [
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
INDEPENDENT_GLOBALS = ["inversion_count", "spearman_footrule", "maximum_rank_error"]
EPISODE_RULES = [
    ("plateau_bridge_initial_included", True, True),
    ("plateau_break_initial_included", False, True),
    ("plateau_bridge_post_swap_only", True, False),
    ("plateau_break_post_swap_only", False, False),
]
STRATA = [
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
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = (
        pa.Table.from_pandas(rows, preserve_index=False)
        if isinstance(rows, pd.DataFrame)
        else pa.Table.from_pylist(list(rows))
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    output = subprocess.check_output(["zstd", "-dc", str(path)])
    return [json.loads(line) for line in output.decode().splitlines() if line]


def stable_id(prefix: str, *parts: object) -> str:
    encoded = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return prefix + hashlib.sha256(encoded).hexdigest()


def category_tertile(progress: pd.Series) -> pd.Series:
    return pd.cut(
        progress,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "middle", "late"],
        include_lowest=True,
    ).astype("string")


def disorder_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=["[0,.2]", "(.2,.4]", "(.4,.6]", "(.6,.8]", "(.8,1]"],
        include_lowest=True,
    ).astype("string")


def scenario_metadata(scenario_ids: Sequence[str]) -> pd.DataFrame:
    reference_ids = sorted(value for value in scenario_ids if value.startswith("r1:"))
    table = pq.read_table(
        SCENARIO_BANK,
        filters=[("scenarioId", "in", reference_ids)],
        columns=[
            "scenarioId", "conditionFamily", "profileRole", "inputProfile", "architecture",
            "policySet", "assignmentProfile", "directionProfile", "faultMode",
            "requestedFaultCount", "realizedFaultCount", "placementProfile", "split",
            "compositionCountsJson", "directionCountsJson",
        ],
    ).to_pandas()
    rows: list[dict[str, Any]] = []
    for row in table.to_dict("records"):
        policies = [str(value) for value in list(row["policySet"])]
        rows.append(
            {
                "scenario_id": row["scenarioId"],
                "condition_family": row["conditionFamily"],
                "profile_role": row["profileRole"],
                "input_profile": row["inputProfile"],
                "architecture": row["architecture"],
                "policy_composition": "+".join(policies),
                "assignment_profile": row["assignmentProfile"],
                "direction_profile": row["directionProfile"],
                "fault_mode": row["faultMode"],
                "requested_fault_count": int(row["requestedFaultCount"]),
                "realized_fault_count": int(row["realizedFaultCount"]),
                "placement_profile": row["placementProfile"],
                "scenario_split": row["split"],
                "composition_counts_json": row["compositionCountsJson"],
                "direction_counts_json": row["directionCountsJson"],
                "scheduler": "serial_counter_addressed",
            }
        )
    rows.extend(
        [
            {
                "scenario_id": "r1:e04316de29fbb3e9c95244bebb0da560584c3990ffc290d5b4e4633310554a20",
                "condition_family": "S06_reference_smoke",
                "profile_role": "reference_validation",
                "input_profile": "unique_n4",
                "architecture": "cell_view",
                "policy_composition": "identity_owned_mixed",
                "assignment_profile": "explicit",
                "direction_profile": "consensus_ascending",
                "fault_mode": "none",
                "requested_fault_count": 0,
                "realized_fault_count": 0,
                "placement_profile": "not_applicable",
                "scenario_split": "smoke",
                "composition_counts_json": None,
                "direction_counts_json": '{"ascending":4}',
                "scheduler": "serial_counter_addressed",
            },
            {
                "scenario_id": "h1:9b02aea2b1251b9843f999ac6afe663ddf14bf65f55ae395931c94a2d85e364b",
                "condition_family": "S06_historical_smoke",
                "profile_role": "supplemental_historical",
                "input_profile": "unique_n2",
                "architecture": "historical_threads",
                "policy_composition": "Bubble",
                "assignment_profile": "pure",
                "direction_profile": "consensus_ascending",
                "fault_mode": "none",
                "requested_fault_count": 0,
                "realized_fault_count": 0,
                "placement_profile": "not_applicable",
                "scenario_split": "smoke",
                "composition_counts_json": None,
                "direction_counts_json": '{"ascending":2}',
                "scheduler": "historical_threads_recorded_swap",
            },
        ]
    )
    result = pd.DataFrame(rows)
    if result["scenario_id"].duplicated().any():
        raise AssertionError("scenario metadata contains duplicate IDs")
    return result


def build_event_atlas(states: pd.DataFrame, diagnostics: Mapping[str, Any]) -> pd.DataFrame:
    keep = states[states["checkpoint_kind"].isin(["initial", "accepted_action"])].copy()
    keep = keep.sort_values(
        ["logical_trace_id", "direction", "activation_count", "accepted_swap_index"]
    ).reset_index(drop=True)
    group_fields = ["logical_trace_id", "direction"]
    shifted_fields = [
        "activation_count", "accepted_swap_index", "native_state_hash", "ordered_values_sha256",
        *DISTANCE_VARIANTS,
    ]
    grouped = keep.groupby(group_fields, sort=False)
    for field in shifted_fields:
        keep[f"previous_{field}"] = grouped[field].shift(1)
    events = keep[keep["checkpoint_kind"] == "accepted_action"].copy()
    if events[[f"previous_{field}" for field in DISTANCE_VARIANTS]].isna().any().any():
        raise AssertionError("an accepted action lacks a predecessor state")
    events["schema_version"] = SCHEMA_VERSION
    events["research_step_id"] = "S03"
    events["neutral_scheduler_opportunities_before_action"] = (
        events["activation_count"] - events["previous_activation_count"] - 1
    ).clip(lower=0).astype("int64")
    events["action_progress"] = events["accepted_swap_index"] / events.groupby(
        group_fields
    )["accepted_swap_index"].transform("max")
    events["action_progress_tertile"] = category_tertile(events["action_progress"])
    initial = (
        keep[keep["checkpoint_kind"] == "initial"]
        .set_index(group_fields)["normalized_kendall_distance"]
        .to_dict()
    )
    events["initial_normalized_kendall_distance"] = [
        initial[(trace, direction)] for trace, direction in zip(events.logical_trace_id, events.direction)
    ]
    events["initial_kendall_disorder_bin"] = disorder_bin(
        events["initial_normalized_kendall_distance"]
    )
    events["pre_action_kendall_disorder_bin"] = disorder_bin(
        events["previous_normalized_kendall_distance"]
    )
    for metric in DISTANCE_VARIANTS:
        events[f"delta_{metric}"] = events[metric] - events[f"previous_{metric}"]
        events[f"sign_{metric}"] = events[f"delta_{metric}"].map(
            lambda value: delta_sign(value, tolerance=TOLERANCE)
        )
    events["delta_paper_sortedness_strict"] = (
        events["paper_sortedness_strict"]
        - (1.0 - events["previous_paper_sortedness_distance"])
    )
    events["progress_sign_paper_sortedness_strict"] = events[
        "delta_paper_sortedness_strict"
    ].map(
        lambda value: (
            "progress" if value > TOLERANCE else "regress" if value < -TOLERANCE else "neutral"
        )
    )
    global_worsening = pd.concat(
        [events[f"sign_{metric}"] == WORSENING for metric in INDEPENDENT_GLOBALS],
        axis=1,
    ).any(axis=1)
    global_improvement = pd.concat(
        [events[f"sign_{metric}"] == IMPROVEMENT for metric in INDEPENDENT_GLOBALS],
        axis=1,
    ).any(axis=1)
    events["paper_backtrack"] = events["sign_paper_sortedness_distance"] == WORSENING
    events["any_independent_global_regression"] = global_worsening
    events["any_independent_global_improvement"] = global_improvement
    events["all_independent_globals_nonworsening"] = ~global_worsening
    events["paper_adjacent_sign_disagreement"] = (
        events["sign_paper_sortedness_distance"] != events["sign_adjacent_descents"]
    )
    events["event_consensus_class"] = np.select(
        [
            events["paper_backtrack"] & ~global_worsening,
            events["paper_backtrack"] & global_worsening,
            ~events["paper_backtrack"] & global_worsening,
        ],
        [
            "paper_backtrack_all_global_nonworsening",
            "paper_backtrack_with_global_regression",
            "global_regression_without_paper_backtrack",
        ],
        default="no_paper_or_global_regression",
    )
    events["is_supplemental_trace"] = events["source_artifact"].str.contains(
        "historical_trace", regex=False
    )
    events["is_primary_release_trace"] = ~events["is_supplemental_trace"]
    events["is_homogeneous_native_goal"] = (
        events["goal_direction_scope"] == "native_homogeneous_goal"
    )
    diagnostic_rows = {row["logicalTraceId"]: row for row in diagnostics["diagnostics"]}
    events["trace_stop_reason"] = events["logical_trace_id"].map(
        lambda value: diagnostic_rows[value]["stopReason"]
    )
    events["trace_accepted_swaps"] = events["logical_trace_id"].map(
        lambda value: diagnostic_rows[value]["acceptedSwaps"]
    )
    events["trace_activation_count"] = events["logical_trace_id"].map(
        lambda value: diagnostic_rows[value]["activationCount"]
    )
    events["stop_reason"] = events["trace_stop_reason"]
    metadata = scenario_metadata(events["scenario_id"].unique().tolist())
    events = events.merge(metadata, on="scenario_id", how="left", validate="many_to_one")
    if events["architecture"].isna().any():
        missing = events.loc[events["architecture"].isna(), "scenario_id"].unique()
        raise AssertionError(f"missing scenario metadata: {missing.tolist()}")
    events["event_id"] = [
        stable_id("s03e:", trace, direction, int(index))
        for trace, direction, index in zip(
            events.logical_trace_id, events.direction, events.accepted_swap_index
        )
    ]
    return events


def build_episodes(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_fields = ["logical_trace_id", "direction"]
    for (trace_id, direction), group in events.groupby(group_fields, sort=True):
        group = group.sort_values("accepted_swap_index").reset_index(drop=True)
        signs = group["sign_paper_sortedness_distance"].tolist()
        for rule_id, bridge_neutral, include_first in EPISODE_RULES:
            boundaries = paper_episode_boundaries(
                signs,
                bridge_neutral=bridge_neutral,
                include_first_transition=include_first,
            )
            for ordinal, boundary in enumerate(boundaries):
                start = group.iloc[boundary.start]
                worsening = group.iloc[boundary.start : boundary.worsening_end + 1]
                recovery = group.iloc[boundary.worsening_end + 1 : boundary.end + 1]
                window = group.iloc[boundary.start : boundary.end + 1]
                paper_worsening = float(
                    worsening.loc[
                        worsening["delta_paper_sortedness_distance"] > TOLERANCE,
                        "delta_paper_sortedness_distance",
                    ].sum()
                )
                paper_recovery = float(
                    -recovery.loc[
                        recovery["delta_paper_sortedness_distance"] < -TOLERANCE,
                        "delta_paper_sortedness_distance",
                    ].sum()
                )
                if paper_worsening <= 0:
                    raise AssertionError("paper episode lacks positive worsening")
                output: dict[str, Any] = {
                    "schema_version": EPISODE_SCHEMA,
                    "research_step_id": "S03",
                    "episode_id": stable_id(
                        "s03p:", trace_id, direction, rule_id, ordinal,
                        int(start["accepted_swap_index"]),
                    ),
                    "boundary_rule": rule_id,
                    "logical_trace_id": trace_id,
                    "scenario_id": start["scenario_id"],
                    "source_step_id": start["source_step_id"],
                    "source_artifact": start["source_artifact"],
                    "direction": direction,
                    "goal_direction_scope": start["goal_direction_scope"],
                    "is_primary_release_trace": bool(start["is_primary_release_trace"]),
                    "is_homogeneous_native_goal": bool(start["is_homogeneous_native_goal"]),
                    "episode_ordinal": ordinal,
                    "start_action_index": int(start["accepted_swap_index"]),
                    "worsening_end_action_index": int(
                        group.iloc[boundary.worsening_end]["accepted_swap_index"]
                    ),
                    "end_action_index": int(group.iloc[boundary.end]["accepted_swap_index"]),
                    "episode_action_span": int(
                        group.iloc[boundary.end]["accepted_swap_index"]
                        - start["accepted_swap_index"]
                        + 1
                    ),
                    "paired_recovery": boundary.paired_recovery,
                    "paper_worsening_magnitude": paper_worsening,
                    "paper_recovery_magnitude": paper_recovery,
                    "paper_recovery_ratio": paper_recovery / paper_worsening,
                    "paper_net_recovery_score": (
                        paper_recovery - paper_worsening
                    ) / paper_worsening,
                    "episode_progress": float(
                        start["accepted_swap_index"] / start["trace_accepted_swaps"]
                    ),
                    "source_step_id": start["source_step_id"],
                    "architecture": start["architecture"],
                    "policy_composition": start["policy_composition"],
                    "condition_family": start["condition_family"],
                    "fault_mode": start["fault_mode"],
                    "realized_fault_count": int(start["realized_fault_count"]),
                    "scheduler": start["scheduler"],
                    "input_profile": start["input_profile"],
                    "stop_reason": start["stop_reason"],
                    "initial_kendall_disorder_bin": start[
                        "initial_kendall_disorder_bin"
                    ],
                }
                any_global_excursion = False
                for metric in DISTANCE_VARIANTS:
                    baseline = float(start[f"previous_{metric}"])
                    values = window[metric].to_numpy(dtype=float)
                    excursion = float(np.max(values - baseline))
                    end_delta = float(values[-1] - baseline)
                    worsening_phase_delta = float(
                        group.iloc[boundary.worsening_end][metric] - baseline
                    )
                    output[f"max_excursion_{metric}"] = excursion
                    output[f"end_delta_{metric}"] = end_delta
                    output[f"paper_worsening_phase_delta_{metric}"] = worsening_phase_delta
                    output[f"excursion_sign_{metric}"] = delta_sign(
                        excursion, tolerance=TOLERANCE
                    )
                    output[f"end_sign_{metric}"] = delta_sign(
                        end_delta, tolerance=TOLERANCE
                    )
                    if metric in INDEPENDENT_GLOBALS and excursion > TOLERANCE:
                        any_global_excursion = True
                output["any_independent_global_excursion"] = any_global_excursion
                output["all_independent_globals_nonworsening"] = not any_global_excursion
                output["episode_consensus_class"] = (
                    "paper_episode_with_global_excursion"
                    if any_global_excursion
                    else "paper_episode_all_global_nonworsening"
                )
                rows.append(output)
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        raise AssertionError("no paper-defined episodes were identified")
    episodes["episode_position_tertile"] = category_tertile(episodes["episode_progress"])
    return episodes


def clustered_metric(
    group: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    label: str,
) -> tuple[float | None, float | None, float | None]:
    trace = (
        group.groupby("logical_trace_id", sort=True)[[numerator, denominator]]
        .sum()
        .to_numpy(dtype=float)
    )
    return clustered_percentile_interval(
        [(float(row[0]), float(row[1])) for row in trace],
        draws=BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, label),
    )


def bootstrap_summaries(events: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    event_primary = events[
        events["is_primary_release_trace"] & events["is_homogeneous_native_goal"]
    ].copy()
    event_primary["denom_action"] = 1
    event_primary["num_paper_backtrack"] = event_primary["paper_backtrack"].astype(int)
    event_primary["denom_paper_backtrack"] = event_primary["paper_backtrack"].astype(int)
    event_primary["num_local_only"] = (
        event_primary["event_consensus_class"]
        == "paper_backtrack_all_global_nonworsening"
    ).astype(int)
    event_primary["num_global_regression"] = event_primary[
        "any_independent_global_regression"
    ].astype(int)
    event_primary["num_adjacent_disagreement"] = event_primary[
        "paper_adjacent_sign_disagreement"
    ].astype(int)
    metrics = [
        ("paper_backtrack_rate", "num_paper_backtrack", "denom_action"),
        (
            "paper_backtrack_all_global_nonworsening_rate",
            "num_local_only",
            "denom_paper_backtrack",
        ),
        ("any_global_regression_rate", "num_global_regression", "denom_action"),
        (
            "paper_adjacent_sign_disagreement_rate",
            "num_adjacent_disagreement",
            "denom_action",
        ),
    ]
    output: list[dict[str, Any]] = []

    def add_event_group(stratum: str, level: str, group: pd.DataFrame) -> None:
        for metric_id, numerator, denominator in metrics:
            estimate, lower, upper = clustered_metric(
                group, numerator, denominator, label=f"event|{stratum}|{level}|{metric_id}"
            )
            output.append(
                {
                    "analysis_unit": "accepted_action",
                    "boundary_rule": None,
                    "stratum": stratum,
                    "level": str(level),
                    "metric": metric_id,
                    "trace_clusters": int(group["logical_trace_id"].nunique()),
                    "denominator": int(group[denominator].sum()),
                    "numerator": int(group[numerator].sum()),
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "logical_trace",
                }
            )

    add_event_group("overall", "all", event_primary)
    for stratum in STRATA:
        for level, group in event_primary.groupby(stratum, dropna=False, sort=True):
            add_event_group(stratum, str(level), group)

    episode_primary = episodes[
        (episodes["boundary_rule"] == "plateau_bridge_initial_included")
        & episodes["is_primary_release_trace"]
        & episodes["is_homogeneous_native_goal"]
    ].copy()
    episode_primary["denom_episode"] = 1
    episode_primary["num_global_nonworsening"] = episode_primary[
        "all_independent_globals_nonworsening"
    ].astype(int)
    episode_primary["num_global_excursion"] = episode_primary[
        "any_independent_global_excursion"
    ].astype(int)
    episode_primary["num_paired"] = episode_primary["paired_recovery"].astype(int)
    episode_primary["denom_paired"] = episode_primary["paired_recovery"].astype(int)
    episode_primary["num_net_positive"] = (
        episode_primary["paired_recovery"]
        & (episode_primary["paper_net_recovery_score"] > TOLERANCE)
    ).astype(int)
    episode_metrics = [
        (
            "paper_episode_all_global_nonworsening_rate",
            "num_global_nonworsening",
            "denom_episode",
        ),
        ("paper_episode_any_global_excursion_rate", "num_global_excursion", "denom_episode"),
        ("paired_recovery_rate", "num_paired", "denom_episode"),
        ("positive_net_recovery_rate_among_paired", "num_net_positive", "denom_paired"),
    ]

    def add_episode_group(stratum: str, level: str, group: pd.DataFrame) -> None:
        for metric_id, numerator, denominator in episode_metrics:
            estimate, lower, upper = clustered_metric(
                group, numerator, denominator, label=f"episode|{stratum}|{level}|{metric_id}"
            )
            output.append(
                {
                    "analysis_unit": "paper_episode",
                    "boundary_rule": "plateau_bridge_initial_included",
                    "stratum": stratum,
                    "level": str(level),
                    "metric": metric_id,
                    "trace_clusters": int(group["logical_trace_id"].nunique()),
                    "denominator": int(group[denominator].sum()),
                    "numerator": int(group[numerator].sum()),
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "logical_trace",
                }
            )

    add_episode_group("overall", "all", episode_primary)
    episode_strata = [value for value in STRATA if value != "action_progress_tertile"] + [
        "episode_position_tertile"
    ]
    for stratum in episode_strata:
        for level, group in episode_primary.groupby(stratum, dropna=False, sort=True):
            add_episode_group(stratum, str(level), group)
    return pd.DataFrame(output)


def sign_summary(events: pd.DataFrame) -> pd.DataFrame:
    populations = {
        "primary_homogeneous": events[
            events["is_primary_release_trace"] & events["is_homogeneous_native_goal"]
        ],
        "mixed_candidate_projections": events[
            events["is_primary_release_trace"] & ~events["is_homogeneous_native_goal"]
        ],
        "supplemental": events[events["is_supplemental_trace"]],
    }
    rows: list[dict[str, Any]] = []
    for population, frame in populations.items():
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
        for stratum in STRATA:
            groups.extend(
                (stratum, str(level), group)
                for level, group in frame.groupby(stratum, dropna=False, sort=True)
            )
        for stratum, level, group in groups:
            paper = group["paper_backtrack"]
            for metric in DISTANCE_VARIANTS:
                for sign in (IMPROVEMENT, NEUTRAL, WORSENING):
                    selected = group[f"sign_{metric}"] == sign
                    rows.append(
                        {
                            "population": population,
                            "stratum": stratum,
                            "level": level,
                            "metric": metric,
                            "sign": sign,
                            "trace_count": int(group["logical_trace_id"].nunique()),
                            "action_count": int(len(group)),
                            "count": int(selected.sum()),
                            "fraction_all_actions": float(selected.mean()) if len(group) else None,
                            "paper_backtrack_count": int(paper.sum()),
                            "count_among_paper_backtracks": int((selected & paper).sum()),
                            "fraction_among_paper_backtracks": (
                                float((selected & paper).sum() / paper.sum())
                                if paper.sum()
                                else None
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def episode_metric_sign_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Summarize every metric's episode excursion/end sign across episode strata."""

    episode_strata = [value for value in STRATA if value != "action_progress_tertile"] + [
        "episode_position_tertile"
    ]
    populations = {
        "primary_homogeneous": episodes[
            episodes["is_primary_release_trace"] & episodes["is_homogeneous_native_goal"]
        ],
        "mixed_candidate_projections": episodes[
            episodes["is_primary_release_trace"] & ~episodes["is_homogeneous_native_goal"]
        ],
        "supplemental": episodes[~episodes["is_primary_release_trace"]],
    }
    rows: list[dict[str, Any]] = []
    for population, frame in populations.items():
        for boundary_rule, boundary_frame in frame.groupby("boundary_rule", sort=True):
            groups: list[tuple[str, str, pd.DataFrame]] = [
                ("overall", "all", boundary_frame)
            ]
            for stratum in episode_strata:
                groups.extend(
                    (stratum, str(level), group)
                    for level, group in boundary_frame.groupby(
                        stratum, dropna=False, sort=True
                    )
                )
            for stratum, level, group in groups:
                for phase in ("excursion", "end"):
                    for metric in DISTANCE_VARIANTS:
                        sign_field = f"{phase}_sign_{metric}"
                        for sign in (IMPROVEMENT, NEUTRAL, WORSENING):
                            selected = group[sign_field] == sign
                            rows.append(
                                {
                                    "population": population,
                                    "boundary_rule": boundary_rule,
                                    "stratum": stratum,
                                    "level": level,
                                    "phase": phase,
                                    "metric": metric,
                                    "sign": sign,
                                    "trace_count": int(group["logical_trace_id"].nunique()),
                                    "episode_count": int(len(group)),
                                    "count": int(selected.sum()),
                                    "fraction": float(selected.mean()) if len(group) else None,
                                }
                            )
    return pd.DataFrame(rows)


def episode_boundary_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = episodes[
        episodes["is_primary_release_trace"] & episodes["is_homogeneous_native_goal"]
    ]
    rows: list[dict[str, Any]] = []
    for rule, group in frame.groupby("boundary_rule", sort=True):
        rows.append(
            {
                "boundary_rule": rule,
                "trace_count": int(group["logical_trace_id"].nunique()),
                "episode_count": int(len(group)),
                "paired_count": int(group["paired_recovery"].sum()),
                "paired_fraction": float(group["paired_recovery"].mean()),
                "all_global_nonworsening_count": int(
                    group["all_independent_globals_nonworsening"].sum()
                ),
                "all_global_nonworsening_fraction": float(
                    group["all_independent_globals_nonworsening"].mean()
                ),
                "positive_net_recovery_count": int(
                    (
                        group["paired_recovery"]
                        & (group["paper_net_recovery_score"] > TOLERANCE)
                    ).sum()
                ),
                "positive_net_recovery_fraction_among_paired": (
                    float(
                        (
                            group["paired_recovery"]
                            & (group["paper_net_recovery_score"] > TOLERANCE)
                        ).sum()
                        / group["paired_recovery"].sum()
                    )
                    if group["paired_recovery"].sum()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def gap_and_denominator_sensitivity(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = [
        row for row in read_zstd_jsonl(GAP_TRACES)
        if row["backendProfile"] == "C-frozen-public-commit"
    ]
    gap_actions = 0
    gap_paper_backtracks = 0
    gap_rows: list[dict[str, Any]] = []
    for row in records:
        scores = [float(point[1]) for point in row["points"]]
        backtracks = sum(right < left - TOLERANCE for left, right in zip(scores, scores[1:]))
        actions = max(len(scores) - 1, 0)
        gap_actions += actions
        gap_paper_backtracks += backtracks
        gap_rows.append(
            {
                "scenarioId": row["scenarioId"],
                "conditionId": row["conditionId"],
                "split": row["split"],
                "policy": row["conditionId"].split("-")[4],
                "acceptedSwapTransitions": actions,
                "paperBacktracksObserved": backtracks,
                "globalClassification": "unavailable_not_imputed",
            }
        )
    primary = events[
        events["is_primary_release_trace"] & events["is_homogeneous_native_goal"]
    ]
    supplemental = events[events["is_homogeneous_native_goal"]]
    mixed = events[~events["is_homogeneous_native_goal"]]
    aligned_actions = len(primary)
    aligned_paper = int(primary["paper_backtrack"].sum())
    aligned_local_only = int(
        (
            primary["event_consensus_class"]
            == "paper_backtrack_all_global_nonworsening"
        ).sum()
    )
    aligned_global = int(primary["any_independent_global_regression"].sum())
    total_actions_with_gap = aligned_actions + gap_actions
    total_paper_with_gap = aligned_paper + gap_paper_backtracks
    rows: list[dict[str, Any]] = []

    def add(label: str, estimate: float | None, lower: float | None, upper: float | None, denominator: int, note: str) -> None:
        rows.append(
            {
                "analysis": label,
                "estimate": estimate,
                "lower_bound": lower,
                "upper_bound": upper,
                "denominator": denominator,
                "note": note,
            }
        )

    add(
        "paper_backtrack_rate_complete_canonical_homogeneous",
        aligned_paper / aligned_actions,
        None,
        None,
        aligned_actions,
        "accepted actions weighted equally; six historical gaps excluded",
    )
    add(
        "paper_backtrack_rate_with_observable_historical_gaps",
        total_paper_with_gap / total_actions_with_gap,
        None,
        None,
        total_actions_with_gap,
        "paper score is directly retained for the six gaps",
    )
    add(
        "local_only_fraction_among_paper_backtracks_complete_case",
        aligned_local_only / aligned_paper if aligned_paper else None,
        None,
        None,
        aligned_paper,
        "global state available",
    )
    add(
        "local_only_fraction_among_paper_backtracks_gap_bounds",
        None,
        aligned_local_only / total_paper_with_gap if total_paper_with_gap else None,
        (aligned_local_only + gap_paper_backtracks) / total_paper_with_gap
        if total_paper_with_gap
        else None,
        total_paper_with_gap,
        "all historical paper backtracks range from globally worsening to nonworsening",
    )
    add(
        "any_global_regression_rate_gap_bounds",
        None,
        aligned_global / total_actions_with_gap,
        (aligned_global + gap_actions) / total_actions_with_gap,
        total_actions_with_gap,
        "all historical accepted actions have unknown global signs",
    )
    trace_rates = primary.groupby("logical_trace_id").agg(
        paper=("paper_backtrack", "mean"),
        local=(
            "event_consensus_class",
            lambda value: float(
                (value == "paper_backtrack_all_global_nonworsening").sum()
                / max((value.isin([
                    "paper_backtrack_all_global_nonworsening",
                    "paper_backtrack_with_global_regression",
                ])).sum(), 1)
            ),
        ),
    )
    add(
        "paper_backtrack_rate_trace_equal",
        float(trace_rates["paper"].mean()),
        None,
        None,
        len(trace_rates),
        "logical traces weighted equally",
    )
    add(
        "local_only_fraction_trace_equal",
        float(trace_rates["local"].mean()),
        None,
        None,
        len(trace_rates),
        "logical traces weighted equally; zero assigned when a trace has no paper backtrack",
    )
    known_activations = int(
        primary.drop_duplicates("logical_trace_id")["trace_activation_count"].fillna(0).sum()
    )
    add(
        "paper_backtrack_rate_per_scheduler_opportunity",
        aligned_paper / known_activations,
        None,
        None,
        known_activations,
        "unchanged scheduler opportunities are neutral; primary homogeneous aligned traces",
    )
    add(
        "paper_backtrack_rate_with_supplemental_s06_historical",
        float(supplemental["paper_backtrack"].mean()),
        None,
        None,
        len(supplemental),
        "supplemental two-cell historical observable trace included",
    )
    for direction, group in mixed.groupby("direction", sort=True):
        add(
            f"mixed_candidate_{direction}_paper_backtrack_rate",
            float(group["paper_backtrack"].mean()),
            None,
            None,
            len(group),
            "paired candidate-goal projection; not part of primary inference",
        )
    gap_detail = {
        "schemaVersion": "e03.s03.coverage_gap_sensitivity.v1",
        "researchStepId": "S03",
        "gapTraceCount": len(records),
        "gapAcceptedSwapTransitions": gap_actions,
        "gapPaperBacktracksObserved": gap_paper_backtracks,
        "traces": gap_rows,
    }
    return pd.DataFrame(rows), gap_detail


def metric_dependence(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(DISTANCE_VARIANTS):
        for right in DISTANCE_VARIANTS[left_index + 1 :]:
            agreement = float((events[f"sign_{left}"] == events[f"sign_{right}"]).mean())
            expected_exact = {left, right} in [
                {"adjacent_descents", "normalized_adjacent_descents"},
                {"inversion_count", "normalized_kendall_distance"},
                {"spearman_footrule", "normalized_spearman_footrule"},
                {"maximum_rank_error", "normalized_maximum_rank_error"},
                {
                    "spearman_footrule",
                    "duplicate_aware_earth_movers_distance",
                },
                {
                    "normalized_spearman_footrule",
                    "normalized_duplicate_aware_earth_movers_distance",
                },
            ]
            rows.append(
                {
                    "metric_left": left,
                    "metric_right": right,
                    "sign_agreement_fraction": agreement,
                    "expected_exact_dependency": expected_exact,
                    "exact_dependency_validated": (agreement == 1.0) if expected_exact else None,
                }
            )
    return pd.DataFrame(rows)


def mixed_projection_sensitivity(events: pd.DataFrame) -> pd.DataFrame:
    mixed = events[~events["is_homogeneous_native_goal"]]
    rows: list[dict[str, Any]] = []
    keys = ["logical_trace_id", "accepted_swap_index"]
    for metric in DISTANCE_VARIANTS:
        pivot = mixed.pivot(index=keys, columns="direction", values=f"sign_{metric}")
        if set(pivot.columns) != {"ascending", "descending"} or pivot.isna().any().any():
            raise AssertionError("mixed-direction action projections do not pair exactly")
        concordant = pivot["ascending"] == pivot["descending"]
        opposite = (
            ((pivot["ascending"] == IMPROVEMENT) & (pivot["descending"] == WORSENING))
            | ((pivot["ascending"] == WORSENING) & (pivot["descending"] == IMPROVEMENT))
        )
        rows.append(
            {
                "metric": metric,
                "paired_actions": len(pivot),
                "concordant_sign_fraction": float(concordant.mean()),
                "opposite_improve_worsen_fraction": float(opposite.mean()),
                "at_least_one_neutral_fraction": float(
                    ((pivot == NEUTRAL).any(axis=1)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def select_representatives(events: pd.DataFrame, episodes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary = events[
        events["is_primary_release_trace"] & events["is_homogeneous_native_goal"]
    ].copy()
    candidates: list[tuple[str, pd.DataFrame, str]] = [
        (
            "paper_local_only",
            primary[
                primary["event_consensus_class"]
                == "paper_backtrack_all_global_nonworsening"
            ],
            "delta_paper_sortedness_distance",
        ),
        (
            "paper_and_global",
            primary[
                primary["event_consensus_class"]
                == "paper_backtrack_with_global_regression"
            ],
            "delta_normalized_kendall_distance",
        ),
        (
            "global_without_paper",
            primary[
                primary["event_consensus_class"]
                == "global_regression_without_paper_backtrack"
            ],
            "delta_normalized_kendall_distance",
        ),
        (
            "tied_local_disagreement",
            primary[
                primary["input_profile"].str.contains("repeated", na=False)
                & primary["paper_adjacent_sign_disagreement"]
            ],
            "delta_paper_sortedness_distance",
        ),
    ]
    selections: list[dict[str, Any]] = []
    anchors: list[tuple[str, pd.Series]] = []
    for category, frame, ranking in candidates:
        if frame.empty:
            selections.append({"category": category, "status": "no_candidate"})
            continue
        chosen = frame.sort_values(
            [ranking, "logical_trace_id", "accepted_swap_index"],
            ascending=[False, True, True],
        ).iloc[0]
        anchors.append((category, chosen))
        selections.append(
            {
                "category": category,
                "status": "selected",
                "eventId": chosen["event_id"],
                "logicalTraceId": chosen["logical_trace_id"],
                "scenarioId": chosen["scenario_id"],
                "direction": chosen["direction"],
                "acceptedSwapIndex": int(chosen["accepted_swap_index"]),
                "eventConsensusClass": chosen["event_consensus_class"],
                "rankingMetric": ranking,
                "rankingValue": float(chosen[ranking]),
            }
        )
    mixed = events[~events["is_homogeneous_native_goal"]]
    pivot = mixed.pivot_table(
        index=["logical_trace_id", "accepted_swap_index"],
        columns="direction",
        values="delta_normalized_kendall_distance",
        aggfunc="first",
    ).dropna()
    if len(pivot):
        pivot["difference"] = (pivot["ascending"] - pivot["descending"]).abs()
        trace, action = pivot.sort_values("difference", ascending=False).index[0]
        chosen = mixed[
            (mixed["logical_trace_id"] == trace)
            & (mixed["accepted_swap_index"] == action)
            & (mixed["direction"] == "ascending")
        ].iloc[0]
        anchors.append(("mixed_projection", chosen))
        selections.append(
            {
                "category": "mixed_projection",
                "status": "selected",
                "eventId": chosen["event_id"],
                "logicalTraceId": trace,
                "scenarioId": chosen["scenario_id"],
                "direction": "paired_ascending_descending",
                "acceptedSwapIndex": int(action),
                "rankingMetric": "absolute_candidate_kendall_delta_difference",
                "rankingValue": float(pivot.loc[(trace, action), "difference"]),
            }
        )
    primary_episodes = episodes[
        (episodes["boundary_rule"] == "plateau_bridge_initial_included")
        & episodes["is_primary_release_trace"]
        & episodes["is_homogeneous_native_goal"]
    ]
    for episode_class in (
        "paper_episode_all_global_nonworsening",
        "paper_episode_with_global_excursion",
    ):
        frame = primary_episodes[
            primary_episodes["episode_consensus_class"] == episode_class
        ]
        category = "episode_" + episode_class.removeprefix("paper_episode_")
        if frame.empty:
            selections.append({"category": category, "status": "no_candidate"})
            continue
        chosen_episode = frame.sort_values(
            ["paper_worsening_magnitude", "logical_trace_id", "start_action_index"],
            ascending=[False, True, True],
        ).iloc[0]
        anchor = events[
            (events["logical_trace_id"] == chosen_episode["logical_trace_id"])
            & (events["direction"] == chosen_episode["direction"])
            & (events["accepted_swap_index"] == chosen_episode["start_action_index"])
        ].iloc[0]
        anchors.append((category, anchor))
        selections.append(
            {
                "category": category,
                "status": "selected",
                "episodeId": chosen_episode["episode_id"],
                "logicalTraceId": chosen_episode["logical_trace_id"],
                "scenarioId": chosen_episode["scenario_id"],
                "direction": chosen_episode["direction"],
                "acceptedSwapIndex": int(chosen_episode["start_action_index"]),
                "rankingMetric": "paper_worsening_magnitude",
                "rankingValue": float(chosen_episode["paper_worsening_magnitude"]),
            }
        )
    windows: list[pd.DataFrame] = []
    for category, anchor in anchors:
        direction_filter = (
            pd.Series(True, index=events.index)
            if category == "mixed_projection"
            else events["direction"] == anchor["direction"]
        )
        frame = events[
            (events["logical_trace_id"] == anchor["logical_trace_id"])
            & direction_filter
            & events["accepted_swap_index"].between(
                max(int(anchor["accepted_swap_index"]) - 10, 1),
                int(anchor["accepted_swap_index"]) + 10,
            )
        ].copy()
        frame.insert(0, "selection_category", category)
        frame.insert(1, "anchor_action_index", int(anchor["accepted_swap_index"]))
        windows.append(frame)
    return pd.concat(windows, ignore_index=True) if windows else pd.DataFrame(), {
        "schemaVersion": "e03.s03.representative_trace_manifest.v1",
        "researchStepId": "S03",
        "selectionRule": "frozen in analysis/s03_atlas_contract.json",
        "selections": selections,
    }


def draw_heatmaps(events: pd.DataFrame, episodes: pd.DataFrame, output: Path) -> None:
    primary = events[
        events["is_primary_release_trace"] & events["is_homogeneous_native_goal"]
        & events["paper_backtrack"]
    ]
    metrics = INDEPENDENT_GLOBALS
    policies = sorted(primary["policy_composition"].unique())
    matrix = np.full((len(policies), len(metrics)), np.nan)
    for i, policy in enumerate(policies):
        group = primary[primary["policy_composition"] == policy]
        for j, metric in enumerate(metrics):
            matrix[i, j] = (group[f"sign_{metric}"] != WORSENING).mean()
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.48 * len(policies) + 1.6)))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(metrics)), ["Kendall", "Footrule/EMD", "Max-rank"])
    ax.set_yticks(range(len(policies)), policies)
    ax.set_title("Global nonworsening fraction among paper-backtrack actions")
    for i in range(len(policies)):
        for j in range(len(metrics)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{100*matrix[i,j]:.1f}%", ha="center", va="center", color="white" if matrix[i,j] < .55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="fraction nonworsening")
    fig.tight_layout()
    fig.savefig(output / "event_sign_heatmap.png", dpi=220)
    fig.savefig(output / "event_sign_heatmap.svg")
    plt.close(fig)

    episode = episodes[
        (episodes["boundary_rule"] == "plateau_bridge_initial_included")
        & episodes["is_primary_release_trace"]
        & episodes["is_homogeneous_native_goal"]
    ]
    sources = sorted(episode["source_step_id"].unique())
    matrix = np.full((len(sources), len(metrics)), np.nan)
    for i, source in enumerate(sources):
        group = episode[episode["source_step_id"] == source]
        for j, metric in enumerate(metrics):
            matrix[i, j] = (group[f"max_excursion_{metric}"] <= TOLERANCE).mean()
    fig, ax = plt.subplots(figsize=(7.8, max(3.5, 0.6 * len(sources) + 1.6)))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(metrics)), ["Kendall", "Footrule/EMD", "Max-rank"])
    ax.set_yticks(range(len(sources)), sources)
    ax.set_title("Paper episodes with no positive global excursion")
    for i in range(len(sources)):
        for j in range(len(metrics)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{100*matrix[i,j]:.1f}%", ha="center", va="center", color="white" if matrix[i,j] < .55 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="fraction without global excursion")
    fig.tight_layout()
    fig.savefig(output / "episode_sign_heatmap.png", dpi=220)
    fig.savefig(output / "episode_sign_heatmap.svg")
    plt.close(fig)


def draw_representatives(windows: pd.DataFrame, output: Path) -> None:
    if windows.empty:
        return
    categories = windows["selection_category"].drop_duplicates().tolist()
    columns = 2
    rows = math.ceil(len(categories) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.5 * rows), squeeze=False)
    metric_styles = [
        ("paper_sortedness_distance", "Paper distance"),
        ("normalized_kendall_distance", "Kendall"),
        ("normalized_spearman_footrule", "Footrule/EMD"),
        ("normalized_maximum_rank_error", "Max-rank"),
    ]
    for ax, category in zip(axes.flat, categories):
        frame = windows[windows["selection_category"] == category].sort_values(
            ["direction", "accepted_swap_index"]
        )
        if category == "mixed_projection":
            for direction, direction_frame in frame.groupby("direction", sort=True):
                ax.plot(
                    direction_frame["accepted_swap_index"],
                    direction_frame["normalized_kendall_distance"],
                    label=f"{direction.capitalize()} Kendall",
                    linewidth=1.7,
                )
                ax.plot(
                    direction_frame["accepted_swap_index"],
                    direction_frame["paper_sortedness_distance"],
                    label=f"{direction.capitalize()} paper",
                    linewidth=1.2,
                    linestyle="--",
                )
            ax.legend(fontsize=8, loc="best")
        else:
            for metric, label in metric_styles:
                ax.plot(
                    frame["accepted_swap_index"], frame[metric], label=label, linewidth=1.5
                )
        anchor = int(frame["anchor_action_index"].iloc[0])
        ax.axvline(anchor, color="black", linestyle="--", linewidth=1)
        ax.set_title(category.replace("_", " "))
        ax.set_xlabel("accepted swap index")
        ax.set_ylabel("normalized distance / proxy")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(categories) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output / "representative_traces.png", dpi=220)
    fig.savefig(output / "representative_traces.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S03"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_paths = [
        S02 / "replayed_distances.parquet",
        S02 / "trace_coverage_manifest.json",
        S02 / "replay_diagnostics.json",
        S02 / "validation_results.json",
        SCENARIO_BANK,
        GAP_TRACES,
        CONTRACT,
    ]
    pre_hash = {str(path): sha256_file(path) for path in input_paths}
    states = pq.read_table(S02 / "replayed_distances.parquet").to_pandas()
    diagnostics = json.loads((S02 / "replay_diagnostics.json").read_text())
    events = build_event_atlas(states, diagnostics)
    episodes = build_episodes(events)
    bootstraps = bootstrap_summaries(events, episodes)
    signs = sign_summary(events)
    episode_signs = episode_metric_sign_summary(episodes)
    boundary = episode_boundary_summary(episodes)
    denominator, gaps = gap_and_denominator_sensitivity(events)
    dependence = metric_dependence(events)
    mixed = mixed_projection_sensitivity(events)
    representatives, representative_manifest = select_representatives(events, episodes)

    write_parquet(args.output / "metric_disagreement_atlas.parquet", events)
    write_parquet(args.output / "metric_disagreement_episodes.parquet", episodes)
    write_parquet(args.output / "metric_sign_summary.parquet", signs)
    write_parquet(args.output / "episode_metric_sign_summary.parquet", episode_signs)
    write_parquet(args.output / "bootstrap_uncertainty.parquet", bootstraps)
    write_parquet(args.output / "representative_traces.parquet", representatives)
    boundary.to_csv(args.output / "episode_boundary_sensitivity.csv", index=False)
    denominator.to_csv(args.output / "denominator_sensitivity.csv", index=False)
    dependence.to_csv(args.output / "metric_dependence.csv", index=False)
    mixed.to_csv(args.output / "mixed_direction_projection_sensitivity.csv", index=False)
    write_json(args.output / "coverage_gap_sensitivity.json", gaps)
    write_json(args.output / "representative_trace_manifest.json", representative_manifest)
    draw_heatmaps(events, episodes, args.output)
    draw_representatives(representatives, args.output)

    post_hash = {str(path): sha256_file(path) for path in input_paths}
    immutable = pre_hash == post_hash
    if not immutable:
        raise AssertionError("one or more S01/S02/E01 inputs changed during S03")
    primary = events[
        events["is_primary_release_trace"] & events["is_homogeneous_native_goal"]
    ]
    primary_episodes = episodes[
        (episodes["boundary_rule"] == "plateau_bridge_initial_included")
        & episodes["is_primary_release_trace"]
        & episodes["is_homogeneous_native_goal"]
    ]
    summary = {
        "schemaVersion": "e03.s03.analysis_summary.v1",
        "researchStepId": "S03",
        "success": True,
        "eventAtlasRows": len(events),
        "alignedLogicalTraces": int(events["logical_trace_id"].nunique()),
        "primaryHomogeneousTraces": int(primary["logical_trace_id"].nunique()),
        "primaryHomogeneousActions": len(primary),
        "paperBacktrackActions": int(primary["paper_backtrack"].sum()),
        "paperBacktrackRate": float(primary["paper_backtrack"].mean()),
        "paperBacktracksAllGlobalNonworsening": int(
            (
                primary["event_consensus_class"]
                == "paper_backtrack_all_global_nonworsening"
            ).sum()
        ),
        "paperBacktracksWithGlobalRegression": int(
            (
                primary["event_consensus_class"]
                == "paper_backtrack_with_global_regression"
            ).sum()
        ),
        "globalRegressionsWithoutPaperBacktrack": int(
            (
                primary["event_consensus_class"]
                == "global_regression_without_paper_backtrack"
            ).sum()
        ),
        "primaryPaperEpisodes": len(primary_episodes),
        "episodesAllGlobalNonworsening": int(
            primary_episodes["all_independent_globals_nonworsening"].sum()
        ),
        "episodesWithGlobalExcursion": int(
            primary_episodes["any_independent_global_excursion"].sum()
        ),
        "gapTraceCount": gaps["gapTraceCount"],
        "gapAcceptedSwapTransitions": gaps["gapAcceptedSwapTransitions"],
        "gapPaperBacktracksObserved": gaps["gapPaperBacktracksObserved"],
        "mixedDirectionLogicalTraces": int(
            events.loc[~events["is_homogeneous_native_goal"], "logical_trace_id"].nunique()
        ),
        "bootstrapDrawsPerEstimate": BOOTSTRAP_DRAWS,
        "inputsUnchanged": immutable,
    }
    write_json(args.output / "analysis_summary.json", summary)
    write_json(
        args.output / "input_immutability.json",
        {
            "schemaVersion": "e03.s03.input_immutability.v1",
            "researchStepId": "S03",
            "success": immutable,
            "preRunSha256": pre_hash,
            "postRunSha256": post_hash,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
