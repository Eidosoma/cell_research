"""Classic-versus-discovered comparison helpers for E03 S13.

S13 compares Bubble, Insertion, and Selection regions against evidence-backed
non-classic DSL policies.  The helpers keep selection, same-seed deltas,
Pareto-front calls, embedding distances, and validation checks deterministic so
the research-step script can write auditable artifacts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .competence import canonical_json


CLASSIC_COMPARISON_VERSION = "e03_s13_classics_vs_discovered.v1"
CLASSIC_FAMILIES = ("bubble", "insertion", "selection")

S13_POLICY_METRICS = (
    "completionScore",
    "sortednessScore",
    "energyScore",
    "robustnessScore",
    "delayedGratificationScore",
    "aggregationScore",
    "oscillationStabilityScore",
)

S13_VECTOR_METRICS = (
    "completionSuccess",
    "finalSortednessScore",
    "energyScore",
    "robustnessScore",
    "delayedGratificationScore",
    "aggregationAucScore",
    "oscillationScore",
)


@dataclass(frozen=True)
class ComparisonTask:
    task_id: str
    task_family: str
    task_panel: str
    input_profile: str
    initial_values: tuple[int, ...]
    scheduler_seed_base: int
    tie_seed_base: int
    frozen_variant: str = "none"
    frozen_positions: tuple[int, ...] = ()
    chimera_with_null: bool = False
    candidate_positions: tuple[int, ...] = ()
    max_activations: int = 640
    max_swaps: int = 640
    max_comparisons: int = 2560

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "inputProfile": self.input_profile,
            "initialValues": list(self.initial_values),
            "schedulerSeedBase": int(self.scheduler_seed_base),
            "tieSeedBase": int(self.tie_seed_base),
            "frozenVariant": self.frozen_variant,
            "frozenPositions": list(self.frozen_positions),
            "chimeraWithNull": bool(self.chimera_with_null),
            "candidatePositions": list(self.candidate_positions),
            "maxActivations": int(self.max_activations),
            "maxSwaps": int(self.max_swaps),
            "maxComparisons": int(self.max_comparisons),
        }


def _safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return default


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda value: bool(value) if pd.notna(value) else False)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _mean_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype=float)
    matrix = pd.concat([pd.to_numeric(df[column], errors="coerce") for column in available], axis=1)
    return matrix.mean(axis=1, skipna=True)


def _sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1))


def classic_family_from_lineage(lineage_id: Any, classic_family: Any = None) -> str:
    """Classify classic lineage strings as bubble, insertion, selection, or none."""

    existing = str(classic_family or "").strip().lower()
    if existing in CLASSIC_FAMILIES:
        return existing
    lineage = str(lineage_id or "").strip().lower()
    for family in CLASSIC_FAMILIES:
        if family in lineage:
            return family
    return "none"


def _selection_reasons(row: pd.Series, s12_source_ids: set[str]) -> list[str]:
    reasons: list[str] = []
    if bool(row.get("isS08Elite", False)):
        reasons.append("s08_elite")
    if bool(row.get("isPhaseBoundaryPolicy", False)):
        reasons.append("s09_phase_boundary")
    if bool(row.get("isPathologicalPolicy", False)):
        reasons.append("s11_pathological_or_failure_mode")
    if pd.notna(row.get("s07_completionSuccessMean", np.nan)):
        reasons.append("s07_measured_competence")
    if pd.notna(row.get("s09_completionSuccessMean", np.nan)) or pd.notna(row.get("s09Trace_completedNumericMean", np.nan)):
        reasons.append("s09_repeated_seed_competence")
    if str(row.get("policyId", "")) in s12_source_ids:
        reasons.append("s12_ablation_source")
    return sorted(set(reasons))


def aggregate_s13_vectors(vector_df: pd.DataFrame, prefix: str = "s13_") -> pd.DataFrame:
    """Aggregate S13 same-seed competence vectors to policy-level metrics."""

    if vector_df.empty:
        return pd.DataFrame(columns=["policyId"])
    df = vector_df.copy()
    df["policyId"] = df["policyId"].astype(str)
    metrics = [column for column in S13_VECTOR_METRICS if column in df.columns]
    for column in metrics:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = df.groupby("policyId", dropna=False)
    base = grouped.agg(
        sameSeedVectorRows=("policyId", "size"),
        sameSeedTaskCount=("taskId", pd.Series.nunique),
        sameSeedTaskFamilyCount=("taskFamily", pd.Series.nunique),
        sameSeedReplicateCount=("replicateIndex", pd.Series.nunique),
    ).reset_index()
    for metric in metrics:
        summary = grouped[metric].agg(["mean", _sd, "min", "max"]).reset_index()
        summary.columns = [
            "policyId" if column == "policyId" else f"{prefix}{metric}{column.capitalize()}"
            for column in summary.columns
        ]
        base = base.merge(summary, on="policyId", how="left")
    if "taskFamily" in df.columns and "completionSuccess" in df.columns:
        pivot = (
            df.pivot_table(index="policyId", columns="taskFamily", values="completionSuccess", aggfunc="mean")
            .add_prefix(f"{prefix}completionRateTaskFamily_")
            .reset_index()
        )
        base = base.merge(pivot, on="policyId", how="left")
    return base


def build_policy_comparison_universe(
    *,
    features_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    s12_policy_specs_df: pd.DataFrame,
    s13_policy_metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return policy-level S13 comparison metadata and broad competence scores."""

    features = features_df.copy()
    features["policyId"] = features["policyId"].astype(str)
    corpus = corpus_df.copy()
    corpus["policyId"] = corpus["policyId"].astype(str)
    corpus_ids = set(corpus["policyId"])
    s12_source_ids = set()
    if not s12_policy_specs_df.empty:
        specs = s12_policy_specs_df.copy()
        if "sourcePolicyId" in specs.columns and "ablationType" in specs.columns:
            s12_source_ids = set(specs.loc[specs["ablationType"].eq("original"), "sourcePolicyId"].astype(str))

    assignment_cols = [
        "policyId",
        "universalityClassId",
        "className",
        "classLabel",
        "classInterpretation",
        "empiricalCaveat",
        "distanceToCentroid",
        "primaryRole",
        "isClassicPolicy",
        "isNullPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "phaseBoundaryInvolvementCount",
        "s08Elite_qualityScore",
    ]
    assignments = assignments_df[[column for column in assignment_cols if column in assignments_df.columns]].copy()
    assignments["policyId"] = assignments["policyId"].astype(str)
    joined = features.merge(assignments, on="policyId", how="left", suffixes=("", "_s11"))
    for column in [
        "primaryRole",
        "isClassicPolicy",
        "isNullPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "phaseBoundaryInvolvementCount",
        "s08Elite_qualityScore",
    ]:
        s11_col = f"{column}_s11"
        if s11_col in joined.columns:
            if column in joined.columns:
                joined[column] = joined[column].combine_first(joined[s11_col])
            else:
                joined[column] = joined[s11_col]
            joined = joined.drop(columns=[s11_col])

    if s13_policy_metrics_df is not None and not s13_policy_metrics_df.empty:
        joined = joined.merge(s13_policy_metrics_df, on="policyId", how="left", validate="one_to_one")

    joined["isStandalonePolicy"] = joined["policyId"].isin(corpus_ids)
    for column in ["isClassicPolicy", "isNullPolicy", "isS08Elite", "isPhaseBoundaryPolicy", "isPathologicalPolicy"]:
        joined[column] = _as_bool(joined.get(column, pd.Series(False, index=joined.index)))
    joined["classicFamily"] = [
        classic_family_from_lineage(lineage, family)
        for lineage, family in zip(joined.get("lineageId", ""), joined.get("classicFamily", ""), strict=False)
    ]

    joined["comparisonSelectionReasons"] = joined.apply(lambda row: _selection_reasons(row, s12_source_ids), axis=1)
    joined["comparisonSelectionReasonsJson"] = joined["comparisonSelectionReasons"].map(canonical_json)
    joined["hasMeasuredCompetence"] = joined["comparisonSelectionReasons"].map(bool)

    def role(row: pd.Series) -> str:
        if bool(row.get("isClassicPolicy", False)):
            return "classic"
        if bool(row.get("isNullPolicy", False)):
            return "null_context"
        if bool(row.get("isStandalonePolicy", False)) and bool(row.get("hasMeasuredCompetence", False)):
            return "discovered"
        return "insufficient_evidence"

    joined["comparisonRole"] = joined.apply(role, axis=1)
    joined["selectedForS13"] = joined["comparisonRole"].isin(["classic", "discovered"])
    joined["completionScore"] = _mean_columns(
        joined,
        ["s07_completionSuccessMean", "s09_completionSuccessMean", "s09Trace_completedNumericMean", "s13_completionSuccessMean"],
    )
    joined["sortednessScore"] = _mean_columns(
        joined,
        ["s07_finalSortednessScoreMean", "s09_finalSortednessScoreMean", "s09Trace_finalSortednessScoreMean", "s13_finalSortednessScoreMean"],
    )
    joined["energyScore"] = _mean_columns(joined, ["s07_energyScoreMean", "s09_energyScoreMean", "s13_energyScoreMean"])
    joined["robustnessScore"] = _mean_columns(joined, ["s07_robustnessScoreMean", "s09_robustnessScoreMean", "s13_robustnessScoreMean"])
    joined["delayedGratificationScore"] = _mean_columns(
        joined, ["s07_delayedGratificationScoreMean", "s09_delayedGratificationScoreMean", "s13_delayedGratificationScoreMean"]
    )
    joined["aggregationScore"] = _mean_columns(joined, ["s07_aggregationAucScoreMean", "s09_aggregationAucScoreMean", "s13_aggregationAucScoreMean"])
    joined["oscillationStabilityScore"] = _mean_columns(joined, ["s07_oscillationScoreMean", "s09_oscillationScoreMean", "s13_oscillationScoreMean"])
    joined["s13CompositeScore"] = joined[list(S13_POLICY_METRICS)].mean(axis=1, skipna=True)
    joined["s13FiniteMetricCount"] = joined[list(S13_POLICY_METRICS)].notna().sum(axis=1).astype(int)
    joined["phaseBoundaryInvolvementCount"] = _numeric(joined, "phaseBoundaryInvolvementCount").fillna(0.0)
    joined["s08Elite_qualityScore"] = _numeric(joined, "s08Elite_qualityScore")
    return joined.sort_values(["comparisonRole", "classicFamily", "s13CompositeScore", "policyId"], ascending=[True, True, False, True], kind="mergesort").reset_index(drop=True)


def pareto_front_table(policy_df: pd.DataFrame, metric_columns: Sequence[str] = S13_POLICY_METRICS) -> pd.DataFrame:
    """Compute non-dominated fronts over higher-is-better policy metrics."""

    df = policy_df[policy_df["selectedForS13"]].copy().reset_index(drop=True)
    metric_columns = [column for column in metric_columns if column in df.columns]
    rows: list[dict[str, Any]] = []
    if not metric_columns or df.empty:
        return pd.DataFrame(columns=["policyId"])
    values = df[metric_columns].to_numpy(dtype=float)
    eligible = np.isfinite(values).all(axis=1)
    ranks = np.full(len(df), -1, dtype=int)
    remaining = set(np.where(eligible)[0].tolist())
    front = 1

    def dominates(left: int, right: int) -> bool:
        return bool(np.all(values[left] >= values[right] - 1e-12) and np.any(values[left] > values[right] + 1e-12))

    while remaining:
        current: list[int] = []
        for index in sorted(remaining):
            if not any(dominates(other, index) for other in remaining if other != index):
                current.append(index)
        for index in current:
            ranks[index] = front
        remaining.difference_update(current)
        front += 1
    for index, row in df.iterrows():
        payload = {
            "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
            "policyId": str(row["policyId"]),
            "comparisonRole": str(row["comparisonRole"]),
            "classicFamily": str(row.get("classicFamily", "none")),
            "primaryRole": str(row.get("primaryRole", "")),
            "paretoEligible": bool(eligible[index]),
            "paretoFrontRank": int(ranks[index]) if ranks[index] > 0 else -1,
            "isParetoOptimal": bool(ranks[index] == 1),
            "s13CompositeScore": float(row["s13CompositeScore"]) if pd.notna(row.get("s13CompositeScore")) else np.nan,
        }
        for metric in metric_columns:
            payload[metric] = float(row[metric]) if pd.notna(row[metric]) else np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def same_seed_task_delta_table(vector_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    """Build classic-discovered same-task, same-seed deltas from S13 vectors."""

    vectors = vector_df.copy()
    vectors["policyId"] = vectors["policyId"].astype(str)
    meta = policy_df[
        [
            "policyId",
            "comparisonRole",
            "classicFamily",
            "primaryRole",
            "className",
            "classLabel",
            "generationMethod",
            "lineageId",
        ]
    ].copy()
    overlapping = [column for column in meta.columns if column != "policyId" and column in vectors.columns]
    if overlapping:
        vectors = vectors.drop(columns=overlapping)
    vectors = vectors.merge(meta, on="policyId", how="left", validate="many_to_one")
    classics = vectors[vectors["comparisonRole"].eq("classic")].copy()
    discovered = vectors[vectors["comparisonRole"].eq("discovered")].copy()
    join_keys = ["taskId", "taskFamily", "taskPanel", "inputProfile", "replicateIndex", "schedulerSeed", "tieBreakerSeed"]
    metrics = [metric for metric in S13_VECTOR_METRICS if metric in vectors.columns]
    classic_cols = join_keys + [
        "policyId",
        "classicFamily",
        "primaryRole",
        "className",
        "classLabel",
        "generationMethod",
        "lineageId",
    ] + metrics
    disc_cols = join_keys + [
        "policyId",
        "primaryRole",
        "className",
        "classLabel",
        "generationMethod",
        "lineageId",
    ] + metrics
    merged = discovered[disc_cols].merge(
        classics[classic_cols],
        on=join_keys,
        how="inner",
        suffixes=("_discovered", "_classic"),
        validate="many_to_many",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        payload = {
            "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
            "classicPolicyId": str(row["policyId_classic"]),
            "discoveredPolicyId": str(row["policyId_discovered"]),
            "classicFamily": str(row["classicFamily"]),
            "classicPrimaryRole": str(row["primaryRole_classic"]),
            "discoveredPrimaryRole": str(row["primaryRole_discovered"]),
            "classicClassName": str(row["className_classic"]),
            "discoveredClassName": str(row["className_discovered"]),
            "classicGenerationMethod": str(row["generationMethod_classic"]),
            "discoveredGenerationMethod": str(row["generationMethod_discovered"]),
            "classicLineageId": str(row["lineageId_classic"]),
            "discoveredLineageId": str(row["lineageId_discovered"]),
            "taskId": str(row["taskId"]),
            "taskFamily": str(row["taskFamily"]),
            "taskPanel": str(row["taskPanel"]),
            "inputProfile": str(row["inputProfile"]),
            "replicateIndex": int(row["replicateIndex"]),
            "schedulerSeed": int(row["schedulerSeed"]),
            "tieBreakerSeed": int(row["tieBreakerSeed"]),
        }
        for metric in metrics:
            classic_value = row[f"{metric}_classic"]
            discovered_value = row[f"{metric}_discovered"]
            payload[f"{metric}Classic"] = float(classic_value) if pd.notna(classic_value) else np.nan
            payload[f"{metric}Discovered"] = float(discovered_value) if pd.notna(discovered_value) else np.nan
            payload[f"{metric}Delta"] = (
                float(discovered_value) - float(classic_value)
                if pd.notna(discovered_value) and pd.notna(classic_value)
                else np.nan
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize_same_seed_pairs(task_delta_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize S13 same-seed deltas over all task families."""

    if task_delta_df.empty:
        return pd.DataFrame()
    group_cols = ["classicPolicyId", "discoveredPolicyId", "classicFamily"]
    metric_delta_cols = [f"{metric}Delta" for metric in S13_VECTOR_METRICS if f"{metric}Delta" in task_delta_df.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in task_delta_df.groupby(group_cols, dropna=False, sort=True):
        classic_id, discovered_id, family = map(str, keys)
        payload: dict[str, Any] = {
            "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
            "classicPolicyId": classic_id,
            "discoveredPolicyId": discovered_id,
            "classicFamily": family,
            "sameSeedPairCount": int(len(group)),
            "sameSeedTaskCount": int(group["taskId"].nunique()),
            "sameSeedTaskFamilyCount": int(group["taskFamily"].nunique()),
            "sameSeedTaskFamiliesJson": canonical_json(sorted(set(map(str, group["taskFamily"])))),
            "sameSeedSchedulerSeedsJson": canonical_json(sorted(set(map(int, group["schedulerSeed"])))),
        }
        dominance_values: list[float] = []
        for delta_col in metric_delta_cols:
            metric = delta_col.removesuffix("Delta")
            deltas = pd.to_numeric(group[delta_col], errors="coerce")
            finite = deltas[np.isfinite(deltas)]
            payload[f"{metric}DeltaMean"] = float(finite.mean()) if not finite.empty else np.nan
            payload[f"{metric}DeltaMedian"] = float(finite.median()) if not finite.empty else np.nan
            payload[f"{metric}ImprovedFraction"] = float((finite > 1e-12).mean()) if not finite.empty else np.nan
            payload[f"{metric}WorsenedFraction"] = float((finite < -1e-12).mean()) if not finite.empty else np.nan
            if not finite.empty:
                dominance_values.append(float(finite.mean()))
        payload["discoveredDominatesClassicSameSeed"] = bool(
            dominance_values and all(value >= -1e-12 for value in dominance_values) and any(value > 1e-12 for value in dominance_values)
        )
        payload["classicDominatesDiscoveredSameSeed"] = bool(
            dominance_values and all(value <= 1e-12 for value in dominance_values) and any(value < -1e-12 for value in dominance_values)
        )
        payload["sameSeedNetAdvantageScore"] = float(np.mean(dominance_values)) if dominance_values else np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def _policy_lookup(policy_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["policyId"]): row.to_dict() for _, row in policy_df.iterrows()}


def _embedding_coordinate_tables(embeddings_df: pd.DataFrame, *, seed: int = 0) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if embeddings_df.empty:
        return tables
    for method, group in embeddings_df[embeddings_df["embeddingSeed"].eq(int(seed))].groupby("embeddingMethod"):
        cols = ["policyId", "embeddingDim1", "embeddingDim2", "embeddingDim3", "embeddingDim4", "embeddingDim5"]
        available = [column for column in cols if column in group.columns]
        tables[str(method)] = group[available].drop_duplicates("policyId").set_index("policyId")
    return tables


def _embedding_distance(tables: Mapping[str, pd.DataFrame], method: str, left: str, right: str) -> float:
    table = tables.get(method)
    if table is None or left not in table.index or right not in table.index:
        return float("nan")
    left_values = pd.to_numeric(table.loc[left], errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(table.loc[right], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if not mask.any():
        return float("nan")
    return float(np.linalg.norm(left_values[mask] - right_values[mask]))


def _normalized_feature_lookup(normalized_matrix_df: pd.DataFrame) -> dict[str, np.ndarray]:
    if normalized_matrix_df.empty or "policyId" not in normalized_matrix_df.columns:
        return {}
    feature_cols = [column for column in normalized_matrix_df.columns if column != "policyId"]
    matrix = normalized_matrix_df.set_index("policyId")[feature_cols].astype(float)
    return {str(policy_id): row.to_numpy(dtype=float) for policy_id, row in matrix.iterrows()}


def _feature_distance(lookup: Mapping[str, np.ndarray], left: str, right: str) -> float:
    if left not in lookup or right not in lookup:
        return float("nan")
    left_values = lookup[left]
    right_values = lookup[right]
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if not mask.any():
        return float("nan")
    return float(np.linalg.norm(left_values[mask] - right_values[mask]))


def _neighbor_rank_lookup(neighbors_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    if neighbors_df.empty:
        return {}
    return {
        (str(row["policyId"]), str(row["neighborPolicyId"])): int(row["neighborRank"])
        for _, row in neighbors_df.iterrows()
    }


def _boundary_pair_lookup(boundaries_df: pd.DataFrame) -> dict[frozenset[str], dict[str, Any]]:
    lookup: dict[frozenset[str], dict[str, Any]] = {}
    if boundaries_df.empty:
        return lookup
    transition_cols = [
        column
        for column in [
            "sortingSuccessTransition",
            "sortednessTransition",
            "delayedGratificationTransition",
            "aggregationTransition",
            "oscillationTransition",
            "robustnessTransition",
        ]
        if column in boundaries_df.columns
    ]
    for _, row in boundaries_df.iterrows():
        left = str(row.get("leftPolicyId", ""))
        right = str(row.get("rightPolicyId", ""))
        if not left or not right:
            continue
        key = frozenset([left, right])
        payload = lookup.setdefault(key, {"count": 0, "kinds": set(), "boundaryIds": []})
        payload["count"] += 1
        payload["boundaryIds"].append(str(row.get("boundaryId", "")))
        for column in transition_cols:
            if bool(row.get(column, False)):
                payload["kinds"].add(column)
    return lookup


def build_pairwise_comparison_table(
    *,
    policy_df: pd.DataFrame,
    same_seed_summary_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    normalized_matrix_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the main S13 classic-vs-discovered pairwise comparison table."""

    selected = policy_df[policy_df["selectedForS13"]].copy()
    classics = selected[selected["comparisonRole"].eq("classic")].copy()
    discovered = selected[selected["comparisonRole"].eq("discovered")].copy()
    policy_meta = _policy_lookup(policy_df)
    pareto_meta = pareto_df.set_index("policyId").to_dict(orient="index") if not pareto_df.empty else {}
    same_seed = (
        same_seed_summary_df.set_index(["classicPolicyId", "discoveredPolicyId"]).to_dict(orient="index")
        if not same_seed_summary_df.empty
        else {}
    )
    feature_lookup = _normalized_feature_lookup(normalized_matrix_df)
    embedding_tables = _embedding_coordinate_tables(embeddings_df, seed=0)
    neighbor_lookup = _neighbor_rank_lookup(neighbors_df)
    boundary_lookup = _boundary_pair_lookup(boundaries_df)

    rows: list[dict[str, Any]] = []
    for _, classic in classics.sort_values(["classicFamily", "policyId"], kind="mergesort").iterrows():
        classic_id = str(classic["policyId"])
        for _, discovered_row in discovered.sort_values(["primaryRole", "policyId"], kind="mergesort").iterrows():
            discovered_id = str(discovered_row["policyId"])
            seed_payload = same_seed.get((classic_id, discovered_id), {})
            boundary_payload = boundary_lookup.get(frozenset([classic_id, discovered_id]), {"count": 0, "kinds": set(), "boundaryIds": []})
            metric_deltas = []
            payload: dict[str, Any] = {
                "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
                "classicPolicyId": classic_id,
                "discoveredPolicyId": discovered_id,
                "classicFamily": str(classic.get("classicFamily", "none")),
                "classicLineageId": str(classic.get("lineageId", "")),
                "discoveredLineageId": str(discovered_row.get("lineageId", "")),
                "discoveredPrimaryRole": str(discovered_row.get("primaryRole", "")),
                "discoveredGenerationMethod": str(discovered_row.get("generationMethod", "")),
                "classicClassName": str(classic.get("className", "")),
                "discoveredClassName": str(discovered_row.get("className", "")),
                "sameUniversalityClass": bool(str(classic.get("className", "")) == str(discovered_row.get("className", ""))),
                "classicDistanceToCentroid": float(classic.get("distanceToCentroid", np.nan))
                if pd.notna(classic.get("distanceToCentroid", np.nan))
                else np.nan,
                "discoveredDistanceToCentroid": float(discovered_row.get("distanceToCentroid", np.nan))
                if pd.notna(discovered_row.get("distanceToCentroid", np.nan))
                else np.nan,
                "classicPhaseBoundaryInvolvementCount": float(classic.get("phaseBoundaryInvolvementCount", 0.0)),
                "discoveredPhaseBoundaryInvolvementCount": float(discovered_row.get("phaseBoundaryInvolvementCount", 0.0)),
                "directPhaseBoundaryCount": int(boundary_payload["count"]),
                "directPhaseBoundaryKindsJson": canonical_json(sorted(boundary_payload["kinds"])),
                "directBoundaryIdsJson": canonical_json(sorted(boundary_payload["boundaryIds"])),
                "normalizedFeatureDistance": _feature_distance(feature_lookup, classic_id, discovered_id),
                "pcaSeed0Distance": _embedding_distance(embedding_tables, "pca", classic_id, discovered_id),
                "spectralSeed0Distance": _embedding_distance(embedding_tables, "spectral", classic_id, discovered_id),
                "mdsSeed0Distance": _embedding_distance(embedding_tables, "mds", classic_id, discovered_id),
                "classicToDiscoveredNeighborRank": neighbor_lookup.get((classic_id, discovered_id), -1),
                "discoveredToClassicNeighborRank": neighbor_lookup.get((discovered_id, classic_id), -1),
            }
            for side, row in [("classic", classic), ("discovered", discovered_row)]:
                pareto = pareto_meta.get(str(row["policyId"]), {})
                payload[f"{side}ParetoFrontRank"] = int(pareto.get("paretoFrontRank", -1))
                payload[f"{side}IsParetoOptimal"] = bool(pareto.get("isParetoOptimal", False))
                payload[f"{side}CompositeScore"] = float(row.get("s13CompositeScore", np.nan)) if pd.notna(row.get("s13CompositeScore", np.nan)) else np.nan
            for metric in S13_POLICY_METRICS:
                classic_value = classic.get(metric, np.nan)
                discovered_value = discovered_row.get(metric, np.nan)
                payload[f"{metric}Classic"] = float(classic_value) if pd.notna(classic_value) else np.nan
                payload[f"{metric}Discovered"] = float(discovered_value) if pd.notna(discovered_value) else np.nan
                payload[f"{metric}Delta"] = (
                    float(discovered_value) - float(classic_value)
                    if pd.notna(discovered_value) and pd.notna(classic_value)
                    else np.nan
                )
                if pd.notna(payload[f"{metric}Delta"]):
                    metric_deltas.append(float(payload[f"{metric}Delta"]))
            payload["discoveredDominatesClassicBroad"] = bool(
                metric_deltas and all(value >= -1e-12 for value in metric_deltas) and any(value > 1e-12 for value in metric_deltas)
            )
            payload["classicDominatesDiscoveredBroad"] = bool(
                metric_deltas and all(value <= 1e-12 for value in metric_deltas) and any(value < -1e-12 for value in metric_deltas)
            )
            for key, value in seed_payload.items():
                payload[key] = value
            rows.append(payload)
    return pd.DataFrame(rows)


def family_summary_table(pairwise_df: pd.DataFrame, policy_df: pd.DataFrame, pareto_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize S13 outcomes by Bubble/Insertion/Selection family."""

    rows: list[dict[str, Any]] = []
    if pairwise_df.empty:
        return pd.DataFrame()
    selected = policy_df[policy_df["selectedForS13"]]
    for family, group in pairwise_df.groupby("classicFamily", dropna=False, sort=True):
        family = str(family)
        classics = selected[selected["comparisonRole"].eq("classic") & selected["classicFamily"].eq(family)]
        discovered_ids = set(map(str, group["discoveredPolicyId"]))
        discovered = selected[selected["policyId"].isin(discovered_ids)]
        sorted_group = group.sort_values(
            ["discoveredDominatesClassicSameSeed", "sameSeedNetAdvantageScore", "sortednessScoreDelta", "completionScoreDelta"],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
        best = sorted_group.iloc[0] if not sorted_group.empty else {}
        family_pareto = pareto_df[pareto_df["policyId"].isin(classics["policyId"].astype(str))]
        rows.append(
            {
                "classicComparisonVersion": CLASSIC_COMPARISON_VERSION,
                "classicFamily": family,
                "classicPolicyCount": int(len(classics)),
                "discoveredPolicyCountCompared": int(len(discovered)),
                "classicParetoOptimalCount": int(family_pareto["isParetoOptimal"].sum()) if "isParetoOptimal" in family_pareto else 0,
                "classicBestCompositeScore": float(pd.to_numeric(classics["s13CompositeScore"], errors="coerce").max()),
                "classicMedianCompositeScore": float(pd.to_numeric(classics["s13CompositeScore"], errors="coerce").median()),
                "discoveredDominatingBroadPairCount": int(group.get("discoveredDominatesClassicBroad", pd.Series(False)).sum()),
                "classicDominatingBroadPairCount": int(group.get("classicDominatesDiscoveredBroad", pd.Series(False)).sum()),
                "discoveredDominatingSameSeedPairCount": int(group.get("discoveredDominatesClassicSameSeed", pd.Series(False)).fillna(False).sum()),
                "classicDominatingSameSeedPairCount": int(group.get("classicDominatesDiscoveredSameSeed", pd.Series(False)).fillna(False).sum()),
                "medianSameSeedNetAdvantageScore": float(pd.to_numeric(group.get("sameSeedNetAdvantageScore", np.nan), errors="coerce").median()),
                "bestDiscoveredPolicyId": str(best.get("discoveredPolicyId", "")),
                "bestDiscoveredPrimaryRole": str(best.get("discoveredPrimaryRole", "")),
                "bestDiscoveredClassName": str(best.get("discoveredClassName", "")),
                "bestDiscoveredSameSeedNetAdvantageScore": float(best.get("sameSeedNetAdvantageScore", np.nan))
                if best is not None and pd.notna(best.get("sameSeedNetAdvantageScore", np.nan))
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def missing_comparison_records(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Return standalone non-classic policies excluded from S13 comparisons."""

    excluded = policy_df[
        policy_df["isStandalonePolicy"]
        & ~policy_df["isClassicPolicy"]
        & ~policy_df["isNullPolicy"]
        & ~policy_df["selectedForS13"]
    ].copy()
    if excluded.empty:
        return pd.DataFrame()
    excluded["missingReason"] = "no_s07_s09_s12_or_elite_phase_boundary_measured_competence"
    return excluded[
        [
            "policyId",
            "family",
            "generationMethod",
            "lineageId",
            "primaryRole",
            "comparisonRole",
            "missingReason",
        ]
    ].reset_index(drop=True)


def validate_classic_comparison_outputs(
    *,
    policy_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    task_delta_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    upstream_statuses: Mapping[str, Mapping[str, Any]],
    repo_test_payload: Mapping[str, Any],
    report_exists: bool,
    figure_paths: Sequence[str],
    s14_dir_exists: bool,
) -> pd.DataFrame:
    """Return S13 validation checks as a table."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append({"checkId": check_id, "success": bool(success), "detail": str(detail)})

    for step in ["S08", "S09", "S10", "S11", "S12"]:
        status = upstream_statuses.get(step, {})
        add(f"upstream_{step.lower()}_success", bool(status.get("success")), f"{step} status={status.get('status')}")

    selected = policy_df[policy_df["selectedForS13"]]
    classics = selected[selected["comparisonRole"].eq("classic")]
    discovered = selected[selected["comparisonRole"].eq("discovered")]
    add(
        "classic_family_coverage",
        set(CLASSIC_FAMILIES).issubset(set(classics["classicFamily"].dropna().astype(str))),
        f"classicFamilies={sorted(set(classics['classicFamily'].dropna().astype(str)))} classicPolicies={len(classics)}",
    )
    add("discovered_policy_coverage", len(discovered) > 0, f"discoveredPolicies={len(discovered)}")
    add("pairwise_rows_complete", len(pairwise_df) == len(classics) * len(discovered), f"pairs={len(pairwise_df)} expected={len(classics) * len(discovered)}")
    add("same_seed_task_deltas", not task_delta_df.empty and task_delta_df["schedulerSeed"].nunique() >= 3, f"taskDeltaRows={len(task_delta_df)} seeds={task_delta_df.get('schedulerSeed', pd.Series(dtype=int)).nunique() if not task_delta_df.empty else 0}")
    add("pareto_front_nonempty", not pareto_df.empty and bool((pareto_df.get("isParetoOptimal", pd.Series(False)) == True).any()), f"paretoRows={len(pareto_df)}")
    add(
        "embedding_distances_present",
        "normalizedFeatureDistance" in pairwise_df.columns and pd.to_numeric(pairwise_df["normalizedFeatureDistance"], errors="coerce").notna().any(),
        "normalized feature distances computed for pairwise comparisons",
    )
    add(
        "phase_boundary_context_present",
        "directPhaseBoundaryCount" in pairwise_df.columns and "classicPhaseBoundaryInvolvementCount" in pairwise_df.columns,
        "direct and aggregate phase-boundary fields present",
    )
    add(
        "class_placements_present",
        "classicClassName" in pairwise_df.columns and "discoveredClassName" in pairwise_df.columns and pairwise_df["classicClassName"].notna().any(),
        "S11 class placement fields present",
    )
    add("family_summary_complete", set(CLASSIC_FAMILIES).issubset(set(family_summary_df.get("classicFamily", pd.Series(dtype=str)).astype(str))), f"families={sorted(set(family_summary_df.get('classicFamily', pd.Series(dtype=str)).astype(str)))}")
    add("comparison_report_exists", bool(report_exists), f"report_exists={report_exists}")
    add("figure_outputs", len([path for path in figure_paths if path]) >= 3, f"figures={len([path for path in figure_paths if path])}")
    add("repo_unit_tests", bool(repo_test_payload.get("success")), f"returnCode={repo_test_payload.get('returnCode')} command={repo_test_payload.get('command')}")
    add("no_s14_artifacts", not s14_dir_exists, f"s14_dir_exists={s14_dir_exists}")
    return pd.DataFrame(rows)
