"""Frontier-candidate selection for E03 S14.

S14 turns the exploratory S07-S13 atlas into a small downstream candidate
library.  The selection is deliberately conservative: classic landmarks are
excluded, source code must be recoverable and hash-stable, and every selected
policy is rechecked under held-out seeds, input-disorder perturbations, and
larger array sizes with the CPU DSL interpreter.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import (
    PolicyRecord,
    _base_row,
    _finalize_row,
    actor_schedule,
    sortedness_metrics,
)
from src.e03.policy_generation import policy_feature_flags, semantic_hash
from src.e03.rule_dsl import DSLArrayState, DSLInterpreter, DSLPolicy, parse_policy, stable_json


FRONTIER_SCHEMA = "eidosoma.e03.frontier_candidates.v1"
DEFAULT_FRONTIER_SEED = 2026070114


@dataclass(frozen=True)
class FrontierConfig:
    """One S14 held-out stress condition."""

    config_id: str
    split: str
    array_size: int
    seed: int
    event_cap: int
    input_profile: str
    scheduler: str = "cyclic_scan_seed_offset"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    out = pd.to_numeric(frame[column], errors="coerce")
    if out.notna().any():
        fill = float(out.median(skipna=True))
    else:
        fill = float(default)
    return out.fillna(fill).astype(float)


def _rank01(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        numeric = numeric.fillna(float(numeric.median(skipna=True)))
    else:
        numeric = numeric.fillna(0.0)
    if len(numeric) <= 1:
        return pd.Series(1.0, index=series.index)
    return numeric.rank(method="average", pct=True, ascending=ascending).astype(float)


def pareto_front_mask(frame: pd.DataFrame, *, maximize: Sequence[str], minimize: Sequence[str] = ()) -> pd.Series:
    """Return a non-dominated mask for numeric objectives.

    NaNs are filled with the column median before dominance testing.  Minimized
    columns are sign-flipped so all comparisons are "higher is better".
    """

    if frame.empty:
        return pd.Series([], dtype=bool, index=frame.index)
    objectives = list(maximize) + list(minimize)
    if not objectives:
        raise ValueError("At least one Pareto objective is required")
    matrix_cols: list[np.ndarray] = []
    for column in maximize:
        matrix_cols.append(_safe_numeric(frame, column).to_numpy(float))
    for column in minimize:
        matrix_cols.append((-_safe_numeric(frame, column)).to_numpy(float))
    matrix = np.stack(matrix_cols, axis=1)
    keep = np.ones(len(frame), dtype=bool)
    for idx in range(len(frame)):
        if not keep[idx]:
            continue
        candidate = matrix[idx]
        dominates_or_ties = np.all(matrix >= candidate, axis=1)
        strictly_better = np.any(matrix > candidate, axis=1)
        dominated_by_other = dominates_or_ties & strictly_better
        dominated_by_other[idx] = False
        if bool(dominated_by_other.any()):
            keep[idx] = False
    return pd.Series(keep, index=frame.index)


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _source_record_from_mapping(record: Mapping[str, Any], *, code_source: str, priority: int) -> dict[str, Any] | None:
    policy_id = record.get("policyId") or record.get("policy_id")
    dsl_source = record.get("dslSource") or record.get("dsl_source")
    if not policy_id or not dsl_source:
        return None
    return {
        "policy_id": str(policy_id),
        "declared_policy_name": str(record.get("policyName") or record.get("policy_name") or ""),
        "declared_source_kind": str(record.get("sourceKind") or record.get("source_kind") or ""),
        "declared_semantic_hash": str(record.get("semanticHash") or record.get("semantic_hash") or ""),
        "declared_dsl_sha256": str(record.get("dslSha256") or record.get("dsl_sha256") or ""),
        "dsl_source": str(dsl_source),
        "code_source": code_source,
        "source_record_schema": str(record.get("schema", "")),
        "source_priority": int(priority),
        "lineage_json": stable_json(
            {
                "generation": record.get("generation"),
                "lineageDepth": record.get("lineageDepth") or record.get("lineage_depth"),
                "parentPolicyIds": record.get("parentPolicyIds") or record.get("parent_policy_ids_json"),
                "mutationOperator": record.get("mutationOperator") or record.get("mutation_operator"),
                "generatorIndex": record.get("generatorIndex") or record.get("generator_index"),
                "generatorSeed": record.get("generatorSeed") or record.get("generator_seed"),
            }
        ),
    }


def build_policy_source_table(
    *,
    generated_policy_library: Path,
    qd_candidates: pd.DataFrame,
    qd_discovered_policy_library: Path | None = None,
) -> pd.DataFrame:
    """Load DSL source rows and verify policy IDs and hashes."""

    rows: list[dict[str, Any]] = []
    for record in _jsonl_records(generated_policy_library):
        row = _source_record_from_mapping(record, code_source=str(generated_policy_library), priority=1)
        if row is not None:
            rows.append(row)
    if qd_discovered_policy_library is not None and qd_discovered_policy_library.exists():
        for record in _jsonl_records(qd_discovered_policy_library):
            row = _source_record_from_mapping(record, code_source=str(qd_discovered_policy_library), priority=2)
            if row is not None:
                rows.append(row)
    if not qd_candidates.empty and "dsl_source" in qd_candidates.columns:
        for record in qd_candidates.to_dict(orient="records"):
            row = _source_record_from_mapping(record, code_source="s08_qd_candidate_summary", priority=3)
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table = (
        table.sort_values(["policy_id", "source_priority"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("policy_id", keep="first")
        .reset_index(drop=True)
    )

    verified_rows: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        record = dict(row)
        try:
            policy = parse_policy(str(record["dsl_source"]))
            computed_sha = policy.sha256
            computed_id = policy.policy_id
            computed_semantic = semantic_hash(policy)
            record.update(
                {
                    "computed_policy_id": computed_id,
                    "computed_dsl_sha256": computed_sha,
                    "computed_semantic_hash": computed_semantic,
                    "computed_policy_name": policy.name,
                    "source_parse_success": True,
                    "source_parse_error": "",
                    "policy_id_matches_source": computed_id == record["policy_id"],
                    "dsl_sha256_matches_source": (not record.get("declared_dsl_sha256")) or computed_sha == record.get("declared_dsl_sha256"),
                    "semantic_hash_matches_source": (not record.get("declared_semantic_hash")) or computed_semantic == record.get("declared_semantic_hash"),
                }
            )
            record["source_hash_valid"] = bool(
                record["policy_id_matches_source"]
                and record["dsl_sha256_matches_source"]
                and record["semantic_hash_matches_source"]
            )
            record.update(policy_feature_flags(policy))
        except Exception as exc:  # pragma: no cover - validation preserves failures.
            record.update(
                {
                    "computed_policy_id": "",
                    "computed_dsl_sha256": "",
                    "computed_semantic_hash": "",
                    "computed_policy_name": "",
                    "source_parse_success": False,
                    "source_parse_error": repr(exc),
                    "policy_id_matches_source": False,
                    "dsl_sha256_matches_source": False,
                    "semantic_hash_matches_source": False,
                    "source_hash_valid": False,
                }
            )
        verified_rows.append(record)
    return pd.DataFrame(verified_rows)


def phase_feature_frame(boundaries: pd.DataFrame, axis_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize S09 phase diagnostics to one row per policy."""

    pieces: list[pd.DataFrame] = []
    if not boundaries.empty:
        boundary = boundaries.copy()
        for column in ("transition_strength", "replication_direction_match", "replication_available"):
            if column in boundary.columns:
                if column.startswith("replication"):
                    boundary[column] = boundary[column].map(_as_bool).astype(float)
                else:
                    boundary[column] = pd.to_numeric(boundary[column], errors="coerce")
        grouped = boundary.groupby("base_policy_id", dropna=False).agg(
            s09_boundary_count=("base_policy_id", "size"),
            s09_max_transition_strength=("transition_strength", "max"),
            s09_mean_transition_strength=("transition_strength", "mean"),
            s09_replication_available_fraction=("replication_available", "mean"),
            s09_replication_direction_match_fraction=("replication_direction_match", "mean"),
        )
        if "boundary_kind" in boundary.columns:
            counts = pd.crosstab(boundary["base_policy_id"], boundary["boundary_kind"])
            counts.columns = [f"s09_boundary_kind_count_{column}" for column in counts.columns]
            grouped = grouped.join(counts, how="left")
        pieces.append(grouped.reset_index().rename(columns={"base_policy_id": "policy_id"}))
    if not axis_summary.empty:
        axis = axis_summary.copy()
        for column in ("mean_final_sortedness", "success_fraction", "dg_fraction", "oscillation_fraction"):
            axis[column] = pd.to_numeric(axis[column], errors="coerce")
        grouped = axis.groupby("base_policy_id", dropna=False).agg(
            s09_axis_count=("axis_name", "nunique"),
            s09_axis_mean_final_sortedness=("mean_final_sortedness", "mean"),
            s09_axis_min_final_sortedness=("mean_final_sortedness", "min"),
            s09_axis_max_final_sortedness=("mean_final_sortedness", "max"),
            s09_axis_mean_success_fraction=("success_fraction", "mean"),
            s09_axis_mean_dg_fraction=("dg_fraction", "mean"),
            s09_axis_mean_oscillation_fraction=("oscillation_fraction", "mean"),
        )
        pieces.append(grouped.reset_index().rename(columns={"base_policy_id": "policy_id"}))
    if not pieces:
        return pd.DataFrame(columns=["policy_id"])
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="policy_id", how="outer")
    return out


def ablation_feature_frame(claims: pd.DataFrame) -> pd.DataFrame:
    """Summarize S12 ablation claims to one row per original policy."""

    if claims.empty:
        return pd.DataFrame(columns=["policy_id"])
    df = claims.copy()
    df["heldout_delta_final_sortedness"] = pd.to_numeric(df.get("heldout_delta_final_sortedness", 0.0), errors="coerce")
    counts = pd.crosstab(df["original_policy_id"], df["claim_type"])
    counts.columns = [f"s12_claim_count_{column}" for column in counts.columns]
    grouped = df.groupby("original_policy_id", dropna=False).agg(
        s12_ablation_claim_count=("claim_type", "size"),
        s12_mean_heldout_delta=("heldout_delta_final_sortedness", "mean"),
        s12_min_heldout_delta=("heldout_delta_final_sortedness", "min"),
        s12_max_heldout_delta=("heldout_delta_final_sortedness", "max"),
    )
    out = grouped.join(counts, how="left").reset_index().rename(columns={"original_policy_id": "policy_id"})
    for column in (
        "s12_claim_count_local_necessary_candidate",
        "s12_claim_count_residual_sufficient_candidate",
        "s12_claim_count_anti_feature_candidate",
        "s12_claim_count_locally_neutral_candidate",
        "s12_claim_count_weak_signal",
    ):
        if column not in out.columns:
            out[column] = 0
    return out


def classic_neighbor_feature_frame(neighbors: pd.DataFrame) -> pd.DataFrame:
    """Summarize S13 nearest-classic-neighbor evidence by discovered policy."""

    if neighbors.empty:
        return pd.DataFrame(columns=["policy_id"])
    df = neighbors.copy()
    df = df[df["neighbor_classic_landmark"].map(lambda value: not _as_bool(value))]
    if df.empty:
        return pd.DataFrame(columns=["policy_id"])
    df["neighbor_rank"] = pd.to_numeric(df["neighbor_rank"], errors="coerce")
    df["neighbor_distance"] = pd.to_numeric(df["neighbor_distance"], errors="coerce")
    grouped = df.groupby("neighbor_policy_id", dropna=False).agg(
        s13_classic_neighbor_hit_count=("target_policy_id", "nunique"),
        s13_best_classic_neighbor_rank=("neighbor_rank", "min"),
        s13_min_classic_neighbor_distance=("neighbor_distance", "min"),
        s13_mean_classic_neighbor_distance=("neighbor_distance", "mean"),
    )
    algorithms = df.groupby("neighbor_policy_id")["target_algorithm"].agg(
        lambda values: stable_json(sorted(set(str(value) for value in values if str(value))))
    )
    out = grouped.join(algorithms.rename("s13_neighbor_algorithms_json"), how="left")
    return out.reset_index().rename(columns={"neighbor_policy_id": "policy_id"})


def add_classic_distance(frame: pd.DataFrame) -> pd.DataFrame:
    """Add embedding distance to nearest embedded classic DSL landmark."""

    out = frame.copy()
    coords = ("embedding_x", "embedding_y", "embedding_z")
    if not set(coords).issubset(out.columns) or "classic_landmark" not in out.columns:
        out["embedding_distance_to_classic"] = np.nan
        return out
    classic = out[out["classic_landmark"].map(_as_bool)]
    if classic.empty:
        out["embedding_distance_to_classic"] = np.nan
        return out
    classic_matrix = classic.loc[:, coords].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    all_matrix = out.loc[:, coords].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    distances = np.sqrt(((all_matrix[:, None, :] - classic_matrix[None, :, :]) ** 2).sum(axis=2))
    out["embedding_distance_to_classic"] = np.nanmin(distances, axis=1)
    return out


def build_frontier_universe(
    *,
    embeddings: pd.DataFrame,
    clusters: pd.DataFrame,
    source_table: pd.DataFrame,
    phase_features: pd.DataFrame,
    ablation_features: pd.DataFrame,
    classic_neighbor_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join S07-S13 evidence into one selectable policy table."""

    base = embeddings.copy()
    cluster_cols = [
        "policy_id",
        "class_id",
        "cautious_label",
        "label_confidence",
        "cluster_distance",
        "usesIdeal",
        "usesPrefixSorted",
        "usesRandomCondition",
        "usesProbabilisticAction",
        "usesMemory",
        "usesSignal",
        "swapActionCount",
        "stateActionCount",
        "waitActionCount",
        "rule_count",
    ]
    base = base.merge(
        clusters[[column for column in cluster_cols if column in clusters.columns]],
        on="policy_id",
        how="left",
        suffixes=("", "_s11"),
        validate="one_to_one",
    )
    base = base.merge(source_table, on="policy_id", how="left", validate="one_to_one")
    for extra in (phase_features, ablation_features, classic_neighbor_features):
        if not extra.empty and "policy_id" in extra.columns:
            base = base.merge(extra, on="policy_id", how="left", validate="one_to_one")
    base = add_classic_distance(base)
    bool_cols = ("classic_landmark", "classic_dsl_seed", "qd_candidate", "s08_archive_winner", "source_hash_valid", "source_parse_success")
    for column in bool_cols:
        if column in base.columns:
            base[column] = base[column].map(_as_bool)
        else:
            base[column] = False
    for column in (
        "s09_boundary_count",
        "s13_classic_neighbor_hit_count",
        "s12_claim_count_local_necessary_candidate",
        "s12_claim_count_residual_sufficient_candidate",
        "s12_claim_count_anti_feature_candidate",
    ):
        if column not in base.columns:
            base[column] = 0
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0).astype(float)
    return add_selection_columns(base)


def add_selection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add frontier labels, eligibility flags, and ranking score."""

    out = frame.copy()
    out["nonclassic_policy"] = ~(out["classic_landmark"].map(_as_bool) | out["classic_dsl_seed"].map(_as_bool))
    out["basic_sorting_prior_pass"] = (
        (_safe_numeric(out, "screen_heldout_final_sortedness_mean") >= 0.45)
        & (_safe_numeric(out, "screen_score") >= 0.45)
        & (_safe_numeric(out, "invalid_run_count") == 0)
    )
    out["candidate_eligible"] = out["nonclassic_policy"] & out["source_hash_valid"].map(_as_bool) & out["basic_sorting_prior_pass"]

    eligible = out[out["candidate_eligible"]].copy()
    frontier_labels = {idx: [] for idx in out.index}
    if not eligible.empty:
        frontier_specs = {
            "screen_pareto": (("screen_heldout_final_sortedness_mean", "screen_heldout_improvement_mean"), ("screen_heldout_work_mean",)),
            "quality_pareto": (("screen_score", "best_final_sortedness"), ("timeout_run_count",)),
            "scale_pareto": (("n100_final_sortedness_mean", "screen_heldout_final_sortedness_mean"), ("screen_heldout_work_mean",)),
            "phase_pareto": (("s09_axis_max_final_sortedness", "s09_replication_direction_match_fraction"), ("s09_axis_mean_oscillation_fraction",)),
            "novelty_pareto": (("screen_score", "embedding_distance_to_classic"), ("cluster_distance",)),
        }
        for label, (maximize, minimize) in frontier_specs.items():
            available_max = [column for column in maximize if column in eligible.columns and eligible[column].notna().any()]
            available_min = [column for column in minimize if column in eligible.columns and eligible[column].notna().any()]
            if not available_max:
                continue
            mask = pareto_front_mask(eligible, maximize=available_max, minimize=available_min)
            for idx in eligible[mask].index:
                frontier_labels[idx].append(label)
    out["pareto_frontier_labels_json"] = [stable_json(labels) for _, labels in sorted(frontier_labels.items(), key=lambda item: item[0])]
    out["pareto_frontier_count"] = [len(frontier_labels[idx]) for idx in out.index]

    score = (
        0.26 * _rank01(out["screen_score"], ascending=True)
        + 0.22 * _rank01(out["screen_heldout_final_sortedness_mean"], ascending=True)
        + 0.13 * _rank01(out["screen_heldout_improvement_mean"], ascending=True)
        + 0.11 * _rank01(out["screen_heldout_work_mean"], ascending=False)
        + 0.08 * _rank01(out.get("n100_final_sortedness_mean", pd.Series(np.nan, index=out.index)), ascending=True)
        + 0.06 * _rank01(out.get("s09_axis_max_final_sortedness", pd.Series(np.nan, index=out.index)), ascending=True)
        + 0.05 * _rank01(out.get("embedding_distance_to_classic", pd.Series(np.nan, index=out.index)), ascending=True)
        + 0.04 * out["pareto_frontier_count"].clip(upper=3).astype(float) / 3.0
        + 0.03 * out.get("s08_archive_winner", pd.Series(False, index=out.index)).map(_as_bool).astype(float)
        + 0.02 * out.get("qd_candidate", pd.Series(False, index=out.index)).map(_as_bool).astype(float)
    )
    score += 0.02 * (_safe_numeric(out, "s12_claim_count_local_necessary_candidate").clip(upper=3) / 3.0)
    score += 0.015 * (_safe_numeric(out, "s12_claim_count_residual_sufficient_candidate").clip(upper=3) / 3.0)
    score -= 0.025 * (_safe_numeric(out, "s12_claim_count_anti_feature_candidate").clip(upper=3) / 3.0)
    out["selection_score"] = score.astype(float)
    return out


def _add_selection(selected: dict[str, dict[str, Any]], row: pd.Series, reason: str) -> None:
    policy_id = str(row["policy_id"])
    if policy_id not in selected:
        record = dict(row)
        record["selection_reasons"] = [reason]
        selected[policy_id] = record
    elif reason not in selected[policy_id]["selection_reasons"]:
        selected[policy_id]["selection_reasons"].append(reason)


def select_frontier_candidates(
    universe: pd.DataFrame,
    *,
    target_count: int = 20,
    final_count: int = 16,
) -> pd.DataFrame:
    """Select a source-backed non-classic frontier/niche candidate pool."""

    if final_count < 10 or final_count > 20:
        raise ValueError("final_count must be in the required 10-20 range")
    eligible = universe[universe["candidate_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible = eligible.sort_values(["selection_score", "screen_score", "policy_id"], ascending=[False, False, True], kind="mergesort")
    selected: dict[str, dict[str, Any]] = {}

    for label in ("screen_pareto", "quality_pareto", "scale_pareto", "phase_pareto", "novelty_pareto"):
        mask = eligible["pareto_frontier_labels_json"].map(lambda text: label in json.loads(str(text)))
        for _, row in eligible[mask].head(3).iterrows():
            _add_selection(selected, row, label)

    for class_id, group in eligible.groupby("class_id", sort=True):
        class_best = group.sort_values(["selection_score", "screen_score"], ascending=[False, False], kind="mergesort").head(1)
        if not class_best.empty:
            _add_selection(selected, class_best.iloc[0], f"niche_best_{class_id}")

    for column, reason in (
        ("s08_archive_winner", "s08_archive_winner"),
        ("qd_candidate", "s08_qd_candidate"),
    ):
        if column in eligible.columns:
            for _, row in eligible[eligible[column].map(_as_bool)].head(5).iterrows():
                _add_selection(selected, row, reason)

    for column, reason in (
        ("s09_boundary_count", "s09_phase_boundary_diagnostic"),
        ("s13_classic_neighbor_hit_count", "s13_classic_neighborhood"),
        ("s12_claim_count_local_necessary_candidate", "s12_local_necessary_signal"),
    ):
        if column in eligible.columns:
            subset = eligible[pd.to_numeric(eligible[column], errors="coerce").fillna(0) > 0]
            for _, row in subset.head(4).iterrows():
                _add_selection(selected, row, reason)

    for source_kind, group in eligible.groupby("source_kind", sort=True):
        if len(selected) >= target_count:
            break
        if str(source_kind) == "classic_dsl_seed":
            continue
        _add_selection(selected, group.iloc[0], f"source_kind_best_{source_kind}")

    for _, row in eligible.iterrows():
        if len(selected) >= target_count:
            break
        _add_selection(selected, row, "score_fill")

    selected_df = pd.DataFrame(selected.values())
    if selected_df.empty:
        return selected_df
    selected_df["selection_reasons_json"] = selected_df["selection_reasons"].map(lambda values: stable_json(sorted(values)))
    selected_df["selection_reason_count"] = selected_df["selection_reasons"].map(len)
    selected_df = selected_df.sort_values(
        ["selection_score", "selection_reason_count", "screen_score", "policy_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).head(target_count)
    selected_df["candidate_pool_rank"] = np.arange(1, len(selected_df) + 1)
    return selected_df.drop(columns=["selection_reasons"]).reset_index(drop=True)


def stress_configs() -> list[FrontierConfig]:
    """Return S14 held-out seed, perturbation, and array-size checks."""

    return [
        FrontierConfig("s14_random_n8_seed9201", "heldout", 8, 9201, 64, "random_permutation"),
        FrontierConfig("s14_random_n8_seed9202", "heldout", 8, 9202, 64, "random_permutation"),
        FrontierConfig("s14_random_n16_seed9301", "heldout", 16, 9301, 128, "random_permutation"),
        FrontierConfig("s14_random_n16_seed9302", "heldout", 16, 9302, 128, "random_permutation"),
        FrontierConfig("s14_nearly_sorted_n16_seed9401", "heldout", 16, 9401, 96, "nearly_sorted"),
        FrontierConfig("s14_reverse_n16_seed9501", "heldout", 16, 9501, 160, "reverse_sorted"),
        FrontierConfig("s14_block_reversed_n16_seed9502", "heldout", 16, 9502, 160, "block_reversed"),
        FrontierConfig("s14_random_n32_seed9601", "heldout", 32, 9601, 256, "random_permutation"),
        FrontierConfig("s14_random_n32_seed9602", "heldout", 32, 9602, 256, "random_permutation"),
        FrontierConfig("s14_nearly_sorted_n32_seed9701", "heldout", 32, 9701, 192, "nearly_sorted"),
        FrontierConfig("s14_block_reversed_n32_seed9702", "heldout", 32, 9702, 256, "block_reversed"),
        FrontierConfig("s14_random_n64_seed9801", "heldout", 64, 9801, 512, "random_permutation"),
    ]


def initial_values_for_profile(array_size: int, seed: int, input_profile: str) -> tuple[int, ...]:
    """Return a deterministic unique-value array for one perturbation profile."""

    values = np.arange(1, array_size + 1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    if input_profile == "random_permutation":
        rng.shuffle(values)
        return tuple(int(value) for value in values)
    if input_profile == "nearly_sorted":
        swaps = max(1, array_size // 8)
        for _ in range(swaps):
            idx = int(rng.integers(0, max(array_size - 1, 1)))
            values[idx], values[idx + 1] = values[idx + 1], values[idx]
        return tuple(int(value) for value in values)
    if input_profile == "reverse_sorted":
        return tuple(int(value) for value in values[::-1])
    if input_profile == "block_reversed":
        block_count = min(4, array_size)
        blocks = np.array_split(values, block_count)
        return tuple(int(value) for block in blocks[::-1] for value in block)
    if input_profile == "alternating_high_low":
        low = 1
        high = array_size
        out: list[int] = []
        while low <= high:
            out.append(low)
            if low != high:
                out.append(high)
            low += 1
            high -= 1
        return tuple(out[:array_size])
    raise ValueError(f"Unsupported S14 input profile: {input_profile}")


def rng_seed_for_policy(policy_id: str, config: FrontierConfig, *, seed: int = DEFAULT_FRONTIER_SEED) -> int:
    digest = hashlib.sha256(f"{seed}|{policy_id}|{config.seed}|{config.config_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def policy_record_from_candidate(row: Mapping[str, Any]) -> PolicyRecord:
    policy = parse_policy(str(row["dsl_source"]))
    return PolicyRecord(
        policy=policy,
        policy_id=str(row["policy_id"]),
        policy_name=str(row.get("policy_name") or row.get("computed_policy_name") or policy.name),
        source_kind=str(row.get("source_kind") or row.get("declared_source_kind") or "s14_candidate"),
        semantic_hash=semantic_hash(policy),
        dsl_sha256=policy.sha256,
        features=policy_feature_flags(policy),
        compiles_for_batch=False,
        requires_cpu_fallback=True,
        fallback_reasons=("s14_cpu_reference_recheck",),
        route="cpu_fallback",
    )


def run_cpu_frontier_config(record: PolicyRecord, config: FrontierConfig, *, seed: int = DEFAULT_FRONTIER_SEED) -> dict[str, Any]:
    """Run one selected policy under one S14 held-out stress config."""

    values = initial_values_for_profile(config.array_size, config.seed, config.input_profile)
    schedule = actor_schedule(config.array_size, config.event_cap, config.seed)
    row = _base_row(
        record,
        # _base_row only needs the config-like attributes used below.
        type(
            "_ConfigAdapter",
            (),
            {
                "config_id": config.config_id,
                "phase": "s14_heldout_recheck",
                "split": config.split,
                "array_size": config.array_size,
                "seed": config.seed,
                "event_cap": config.event_cap,
                "input_profile": config.input_profile,
                "scheduler": config.scheduler,
            },
        )(),
        values,
    )
    row.update({"schema": FRONTIER_SCHEMA, "research_step_id": "S14", "s14_rng_seed": rng_seed_for_policy(record.policy_id, config, seed=seed)})
    started = time.perf_counter()
    try:
        interpreter = DSLInterpreter(record.policy)
        rng = random.Random(int(row["s14_rng_seed"]))
        statuses: tuple[str, ...] = tuple("ACTIVE" for _ in values)
        ideal_position: int | None = None
        compare_count = 0
        swap_count = 0
        update_count = 0
        wait_count = 0
        current_values = values
        for actor_index in schedule:
            state = DSLArrayState(
                values=current_values,
                statuses=statuses,
                actor_index=int(actor_index),
                ideal_position=ideal_position,
            )
            result = interpreter.step_state(state, rng)
            current_values = result.state_after.values
            statuses = tuple(result.state_after.statuses or statuses)
            ideal_position = result.state_after.ideal_position
            compare_count += int(bool(result.action.compare_counted))
            swap_count += int(result.action.action_type == "swap")
            update_count += int(result.action.action_type == "update_state")
            wait_count += int(result.action.action_type == "wait")
        elapsed = time.perf_counter() - started
        return _finalize_row(
            row,
            final_values=current_values,
            compare_count=compare_count,
            swap_count=swap_count,
            update_count=update_count,
            wait_count=wait_count,
            elapsed_seconds=elapsed,
            backend="cpu_s14",
            device="cpu",
        )
    except Exception as exc:  # pragma: no cover - preserved in validation artifacts.
        elapsed = time.perf_counter() - started
        return _finalize_row(
            row,
            final_values=values,
            compare_count=0,
            swap_count=0,
            update_count=0,
            wait_count=0,
            elapsed_seconds=elapsed,
            execution_status="invalid",
            error_message=repr(exc),
            backend="cpu_s14",
            device="cpu",
        )


def evaluate_candidate_pool(candidates: pd.DataFrame, configs: Sequence[FrontierConfig], *, seed: int = DEFAULT_FRONTIER_SEED) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict(orient="records"):
        record = policy_record_from_candidate(candidate)
        for config in configs:
            row = run_cpu_frontier_config(record, config, seed=seed)
            row.update(
                {
                    "candidate_pool_rank": int(candidate.get("candidate_pool_rank", 0)),
                    "candidate_selection_score": float(candidate.get("selection_score", np.nan)),
                    "selection_reasons_json": str(candidate.get("selection_reasons_json", "[]")),
                    "class_id": str(candidate.get("class_id", "")),
                    "cautious_label": str(candidate.get("cautious_label", "")),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def recheck_summary_frame(run_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate S14 re-evaluation rows and merge into candidate metadata."""

    if run_df.empty:
        return pd.DataFrame()
    df = run_df.copy()
    df["final_inversion_sortedness"] = pd.to_numeric(df["final_inversion_sortedness"], errors="coerce")
    df["inversion_sortedness_delta"] = pd.to_numeric(df["inversion_sortedness_delta"], errors="coerce")
    df["work_count"] = pd.to_numeric(df["work_count"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for policy_id, group in df.groupby("policy_id", sort=False):
        random_group = group[group["input_profile"] == "random_permutation"]
        perturb_group = group[group["input_profile"] != "random_permutation"]
        n32_plus = group[group["array_size"] >= 32]
        summary = {
            "policy_id": str(policy_id),
            "s14_recheck_run_count": int(len(group)),
            "s14_profile_count": int(group["input_profile"].nunique()),
            "s14_array_size_count": int(group["array_size"].nunique()),
            "s14_invalid_run_count": int(group["invalid"].sum()),
            "s14_timeout_run_count": int(group["timed_out"].sum()),
            "s14_mean_final_sortedness": float(group["final_inversion_sortedness"].mean()),
            "s14_min_final_sortedness": float(group["final_inversion_sortedness"].min()),
            "s14_random_mean_final_sortedness": float(random_group["final_inversion_sortedness"].mean()) if not random_group.empty else np.nan,
            "s14_perturbation_mean_final_sortedness": float(perturb_group["final_inversion_sortedness"].mean()) if not perturb_group.empty else np.nan,
            "s14_n32plus_mean_final_sortedness": float(n32_plus["final_inversion_sortedness"].mean()) if not n32_plus.empty else np.nan,
            "s14_mean_improvement": float(group["inversion_sortedness_delta"].mean()),
            "s14_sorted_run_fraction": float(group["final_is_sorted"].mean()),
            "s14_timeout_fraction": float(group["timed_out"].mean()),
            "s14_mean_work": float(group["work_count"].mean()),
        }
        rows.append(summary)
    summary_df = pd.DataFrame(rows)
    merged = candidates.merge(summary_df, on="policy_id", how="left", validate="one_to_one")
    prior = pd.to_numeric(
        merged["screen_heldout_final_sortedness_mean"]
        if "screen_heldout_final_sortedness_mean" in merged.columns
        else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    )
    random_mean = pd.to_numeric(merged["s14_random_mean_final_sortedness"], errors="coerce")
    merged["s14_retention_ratio_vs_s07"] = random_mean / prior.replace(0.0, np.nan)
    merged["s14_basic_sorting_retained"] = (
        (pd.to_numeric(merged["s14_invalid_run_count"], errors="coerce").fillna(1) == 0)
        & (pd.to_numeric(merged["s14_mean_final_sortedness"], errors="coerce").fillna(0.0) >= 0.50)
        & (pd.to_numeric(merged["s14_mean_improvement"], errors="coerce").fillna(-1.0) >= -0.02)
    )
    merged["s14_stress_retained"] = (
        merged["s14_basic_sorting_retained"]
        & (pd.to_numeric(merged["s14_perturbation_mean_final_sortedness"], errors="coerce").fillna(0.0) >= 0.45)
        & (pd.to_numeric(merged["s14_n32plus_mean_final_sortedness"], errors="coerce").fillna(0.0) >= 0.45)
        & (pd.to_numeric(merged["s14_retention_ratio_vs_s07"], errors="coerce").fillna(0.0) >= 0.65)
    )
    merged["s14_candidate_status"] = np.where(
        merged["s14_stress_retained"],
        "validated_frontier_candidate",
        np.where(merged["s14_basic_sorting_retained"], "stress_fragile_candidate", "failed_heldout_candidate"),
    )
    return merged


def finalize_candidate_set(
    rechecked: pd.DataFrame,
    *,
    final_count: int = 16,
    min_count: int = 10,
) -> pd.DataFrame:
    """Pick the final 10-20 candidate library after S14 rechecks."""

    if rechecked.empty:
        return rechecked.copy()
    status_priority = {"validated_frontier_candidate": 0, "stress_fragile_candidate": 1, "failed_heldout_candidate": 2}
    sortable = rechecked.copy()
    for column in ("selection_score", "s14_mean_final_sortedness", "screen_score"):
        if column not in sortable.columns:
            sortable[column] = np.nan
    sortable["_status_priority"] = sortable["s14_candidate_status"].map(lambda value: status_priority.get(str(value), 99))
    validated = sortable[sortable["s14_candidate_status"] == "validated_frontier_candidate"].copy()
    if len(validated) >= min_count:
        ranked_validated = validated.sort_values(
            ["selection_score", "s14_mean_final_sortedness", "screen_score", "policy_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        pieces: list[pd.DataFrame] = []
        for _, group in ranked_validated.groupby("class_id", sort=True):
            pieces.append(group.head(1))
        diverse = pd.concat(pieces, ignore_index=False) if pieces else ranked_validated.head(0)
        fill = ranked_validated[~ranked_validated["policy_id"].isin(set(diverse["policy_id"]))]
        final = pd.concat([diverse, fill], ignore_index=False).head(final_count)
        final = final.sort_values(
            ["selection_score", "s14_mean_final_sortedness", "screen_score", "policy_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
    else:
        final = sortable.sort_values(
            ["_status_priority", "selection_score", "s14_mean_final_sortedness", "screen_score", "policy_id"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        ).head(max(min_count, min(final_count, len(sortable))))
    final = final.drop(columns=["_status_priority"]).reset_index(drop=True)
    final["curated_rank"] = np.arange(1, len(final) + 1)
    return final


def candidate_policy_json_records(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSONL-ready records for the curated candidate policy library."""

    records: list[dict[str, Any]] = []
    for row in candidates.sort_values("curated_rank").to_dict(orient="records"):
        policy = parse_policy(str(row["dsl_source"]))
        records.append(
            {
                "schema": FRONTIER_SCHEMA,
                "experimentId": "E03",
                "researchStepId": "S14",
                "curatedRank": int(row["curated_rank"]),
                "policyId": str(row["policy_id"]),
                "policyName": str(row["policy_name"]),
                "sourceKind": str(row["source_kind"]),
                "classId": str(row.get("class_id", "")),
                "cautiousLabel": str(row.get("cautious_label", "")),
                "selectionScore": _safe_float(row.get("selection_score"), np.nan),
                "s14CandidateStatus": str(row.get("s14_candidate_status", "")),
                "s14MeanFinalSortedness": _safe_float(row.get("s14_mean_final_sortedness"), np.nan),
                "s14PerturbationMeanFinalSortedness": _safe_float(row.get("s14_perturbation_mean_final_sortedness"), np.nan),
                "s14RetentionRatioVsS07": _safe_float(row.get("s14_retention_ratio_vs_s07"), np.nan),
                "selectionReasons": json.loads(str(row.get("selection_reasons_json", "[]"))),
                "paretoFrontiers": json.loads(str(row.get("pareto_frontier_labels_json", "[]"))),
                "dslSha256": policy.sha256,
                "semanticHash": semantic_hash(policy),
                "sourceHashVerified": bool(row.get("source_hash_valid")),
                "sourceCodePath": str(row.get("code_source", "")),
                "dslSource": policy.to_source(),
            }
        )
    return records


def frontier_candidate_digest(candidates: pd.DataFrame) -> str:
    cols = [
        "curated_rank",
        "policy_id",
        "computed_dsl_sha256",
        "selection_score",
        "s14_mean_final_sortedness",
        "s14_candidate_status",
        "selection_reasons_json",
    ]
    available = [column for column in cols if column in candidates.columns]
    payload: list[dict[str, Any]] = []
    for row in candidates[available].sort_values("curated_rank").to_dict(orient="records"):
        payload.append(
            {
                key: (round(float(value), 12) if isinstance(value, float) and not pd.isna(value) else value)
                for key, value in row.items()
            }
        )
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def validation_frame(
    *,
    universe: pd.DataFrame,
    selected_pool: pd.DataFrame,
    recheck_runs: pd.DataFrame,
    rechecked_pool: pd.DataFrame,
    final_candidates: pd.DataFrame,
    json_records: Sequence[Mapping[str, Any]],
    profiles_written: bool,
    figure_written: bool,
    unit_success: bool,
    target_count: int,
    config_count: int,
) -> pd.DataFrame:
    cases: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append(
            {
                "validation_case": name,
                "success": bool(success),
                "expected": expected,
                "observed": str(observed),
                "notes": notes,
            }
        )

    add("eligible_source_backed_universe", int(universe["candidate_eligible"].sum()) >= target_count, f">={target_count}", int(universe["candidate_eligible"].sum()), "Enough non-classic source-backed policies were available.")
    add("candidate_pool_size", len(selected_pool) >= target_count, f">={target_count}", len(selected_pool), "A pre-recheck pool was selected from frontier and niche evidence.")
    add("final_candidate_count_range", 10 <= len(final_candidates) <= 20, "10-20 final candidates", len(final_candidates), "S14 curated library size is within the requested range.")
    add("final_candidates_nonclassic", not final_candidates.empty and bool(final_candidates["nonclassic_policy"].all()), "all non-classic", final_candidates[["policy_id", "classic_landmark", "classic_dsl_seed"]].to_dict(orient="records"), "Classic DSL landmarks are excluded from the candidate library.")
    add("final_sources_hash_verified", not final_candidates.empty and bool(final_candidates["source_hash_valid"].all()), "all source hashes valid", int(final_candidates["source_hash_valid"].sum()), "Parsed policy ID, DSL SHA-256, and semantic hash match source records.")
    add("distinct_niches_present", final_candidates["class_id"].nunique() >= 4, ">=4 S11 classes", final_candidates["class_id"].nunique(), "Candidate set covers multiple S11 behavior niches.")
    add(
        "pareto_frontier_evidence_present",
        int((final_candidates["pareto_frontier_count"] > 0).sum()) >= max(5, len(final_candidates) // 2),
        "at least half on Pareto fronts",
        int((final_candidates["pareto_frontier_count"] > 0).sum()),
        "Most candidates have direct Pareto-frontier membership; other entries are niche fills.",
    )
    add("recheck_row_count", len(recheck_runs) == len(selected_pool) * config_count, f"{len(selected_pool) * config_count}", len(recheck_runs), "Every selected pool policy was run on every S14 held-out config.")
    add("heldout_profiles_and_sizes", recheck_runs["input_profile"].nunique() >= 4 and recheck_runs["array_size"].nunique() >= 4, ">=4 profiles and >=4 sizes", {"profiles": recheck_runs["input_profile"].nunique(), "sizes": recheck_runs["array_size"].nunique()}, "Recheck covers seeds, perturbations, and array sizes.")
    add("invalid_runs_classified", "invalid" in recheck_runs.columns and "run_status" in recheck_runs.columns, "invalid/run_status columns", list(recheck_runs.columns), "Invalid and timeout labels are explicit.")
    add("final_candidates_retain_basic_sorting", not final_candidates.empty and bool(final_candidates["s14_basic_sorting_retained"].all()), "all final candidates retain basic sorting", final_candidates[["policy_id", "s14_candidate_status"]].to_dict(orient="records"), "Curated candidates retain basic sorting under held-out recheck thresholds.")
    add("policy_jsonl_matches_final", len(json_records) == len(final_candidates), "one JSON record per final candidate", len(json_records), "The downstream JSONL library mirrors the curated CSV.")
    add("profiles_report_written", profiles_written, "profiles markdown exists", profiles_written, "Candidate profiles report was written.")
    add("figure_written", figure_written, "figure exists", figure_written, "Compact validation figure was written.")
    add("unit_tests_passed", bool(unit_success), "unit tests pass", unit_success, "S14 script ran the E03 unit suite.")
    return pd.DataFrame(cases)
