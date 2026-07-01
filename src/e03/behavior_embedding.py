"""Behavior embedding utilities for E03 S10.

S10 embeds policies by measured behavior: S07/S08 competence summaries plus
S09 trajectory and phase diagnostics.  Policy grammar features and source-code
similarity are kept as metadata, not embedding coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


EMBEDDING_SCHEMA = "eidosoma.e03.policy_behavior_embedding.v1"
DEFAULT_EMBEDDING_SEED = 2026070110
DEFAULT_STABILITY_SEEDS = (2026070110, 2026070111, 2026070112, 2026070113, 2026070114)

COMPETENCE_FEATURE_COLUMNS = (
    "screen_run_count",
    "heldout_run_count",
    "scale_run_count",
    "invalid_run_count",
    "timeout_run_count",
    "screen_train_final_sortedness_mean",
    "screen_heldout_final_sortedness_mean",
    "screen_heldout_sorted_run_fraction",
    "screen_heldout_improvement_mean",
    "screen_heldout_work_mean",
    "screen_score",
    "n100_final_sortedness_mean",
    "n1000_final_sortedness_mean",
    "best_final_sortedness",
    "quality_score",
)

METADATA_COLUMNS = (
    "schema",
    "experiment_id",
    "research_step_id",
    "policy_id",
    "policy_name",
    "source_kind",
    "route",
    "requires_cpu_fallback",
    "classic_dsl_seed",
    "embedding_input_source",
    "qd_candidate",
    "s08_archive_winner",
    "classic_family",
    "classic_landmark",
    "s09_diagnostic_available",
)


@dataclass(frozen=True)
class NormalizedFeatureMatrix:
    """A finite numeric feature matrix and provenance for its columns."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    imputed_feature_names: tuple[str, ...]
    missing_fraction: np.ndarray
    stats: pd.DataFrame
    method: str


@dataclass(frozen=True)
class EmbeddingResult:
    """PCA embedding plus normalized input details."""

    embedding: np.ndarray
    explained_variance_ratio: tuple[float, float, float]
    normalized: NormalizedFeatureMatrix
    pca: PCA


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _safe_bool_float(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[column].map(_as_bool).astype(float)


def _first_present(row: pd.Series, columns: Sequence[str], default: Any = np.nan) -> Any:
    for column in columns:
        if column in row.index and not pd.isna(row[column]):
            return row[column]
    return default


def classic_family(policy_name: Any, classic_flag: Any = False) -> str:
    """Return the classic family label used for landmark highlighting."""

    if not _as_bool(classic_flag):
        return ""
    name = str(policy_name).lower()
    if "bubble" in name:
        return "Bubble"
    if "insertion" in name:
        return "Insertion"
    if "selection" in name:
        return "Selection"
    return "Classic"


def aggregate_phase_diagnostics(phase_sweeps: pd.DataFrame, boundaries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate S09 row-level phase diagnostics to one row per base policy."""

    if phase_sweeps.empty:
        return pd.DataFrame(columns=["policy_id", "s09_run_count"])

    df = phase_sweeps.copy()
    required = {"base_policy_id", "axis_name", "failure_mode"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing S09 phase diagnostic columns: {missing}")

    for column in (
        "final_inversion_sortedness",
        "inversion_sortedness_delta",
        "max_inversion_sortedness",
        "min_inversion_sortedness",
        "dg_drop",
        "oscillation_score",
        "no_change_event_fraction",
        "distinct_state_fraction",
        "work_count",
        "event_cap",
        "array_size",
        "axis_numeric",
    ):
        df[column] = _safe_numeric(df, column)
    for column in ("dg_present", "competent_at_threshold", "final_is_sorted", "timed_out", "cycle_detected"):
        df[column] = _safe_bool_float(df, column)

    grouped = df.groupby("base_policy_id", dropna=False)
    general = grouped.agg(
        s09_run_count=("base_policy_id", "size"),
        s09_axis_count=("axis_name", "nunique"),
        s09_mean_final_sortedness=("final_inversion_sortedness", "mean"),
        s09_max_final_sortedness=("final_inversion_sortedness", "max"),
        s09_min_final_sortedness=("final_inversion_sortedness", "min"),
        s09_mean_sortedness_delta=("inversion_sortedness_delta", "mean"),
        s09_max_sortedness_delta=("inversion_sortedness_delta", "max"),
        s09_mean_work=("work_count", "mean"),
        s09_mean_event_cap=("event_cap", "mean"),
        s09_mean_array_size=("array_size", "mean"),
        s09_competent_fraction=("competent_at_threshold", "mean"),
        s09_final_sorted_fraction=("final_is_sorted", "mean"),
        s09_timeout_fraction=("timed_out", "mean"),
        s09_dg_fraction=("dg_present", "mean"),
        s09_mean_dg_drop=("dg_drop", "mean"),
        s09_cycle_fraction=("cycle_detected", "mean"),
        s09_mean_oscillation_score=("oscillation_score", "mean"),
        s09_mean_no_change_fraction=("no_change_event_fraction", "mean"),
        s09_mean_distinct_state_fraction=("distinct_state_fraction", "mean"),
    )

    failure = pd.crosstab(df["base_policy_id"], df["failure_mode"], normalize="index")
    failure.columns = [f"s09_failure_fraction_{str(column)}" for column in failure.columns]

    axis_metrics = (
        df.groupby(["base_policy_id", "axis_name"], dropna=False)
        .agg(
            mean_final_sortedness=("final_inversion_sortedness", "mean"),
            success_fraction=("competent_at_threshold", "mean"),
            dg_fraction=("dg_present", "mean"),
            oscillation_fraction=("cycle_detected", "mean"),
            mean_no_change_fraction=("no_change_event_fraction", "mean"),
        )
        .reset_index()
    )
    axis_pieces: list[pd.DataFrame] = []
    for metric in (
        "mean_final_sortedness",
        "success_fraction",
        "dg_fraction",
        "oscillation_fraction",
        "mean_no_change_fraction",
    ):
        pivot = axis_metrics.pivot(index="base_policy_id", columns="axis_name", values=metric)
        pivot.columns = [f"s09_axis_{str(axis)}_{metric}" for axis in pivot.columns]
        axis_pieces.append(pivot)

    out = general.join(failure, how="left")
    for piece in axis_pieces:
        out = out.join(piece, how="left")

    if boundaries is not None and not boundaries.empty:
        boundary = boundaries.copy()
        for column in (
            "transition_strength",
            "replication_direction_match",
            "replication_available",
            "lower_success_fraction",
            "upper_success_fraction",
            "lower_mean_final_sortedness",
            "upper_mean_final_sortedness",
        ):
            if column in boundary.columns:
                if column.startswith("replication"):
                    boundary[column] = boundary[column].map(_as_bool).astype(float)
                else:
                    boundary[column] = pd.to_numeric(boundary[column], errors="coerce")
        bgroup = boundary.groupby("base_policy_id", dropna=False).agg(
            s09_boundary_count=("base_policy_id", "size"),
            s09_boundary_mean_transition_strength=("transition_strength", "mean"),
            s09_boundary_max_transition_strength=("transition_strength", "max"),
            s09_boundary_replicated_fraction=("replication_available", "mean"),
            s09_boundary_direction_match_fraction=("replication_direction_match", "mean"),
            s09_boundary_lower_success_mean=("lower_success_fraction", "mean"),
            s09_boundary_upper_success_mean=("upper_success_fraction", "mean"),
            s09_boundary_lower_sortedness_mean=("lower_mean_final_sortedness", "mean"),
            s09_boundary_upper_sortedness_mean=("upper_mean_final_sortedness", "mean"),
        )
        if "boundary_kind" in boundary.columns:
            bpivot = pd.crosstab(boundary["base_policy_id"], boundary["boundary_kind"])
            bpivot.columns = [f"s09_boundary_kind_count_{str(column)}" for column in bpivot.columns]
            bgroup = bgroup.join(bpivot, how="left")
        out = out.join(bgroup, how="left")

    out = out.reset_index().rename(columns={"base_policy_id": "policy_id"})
    return out


def build_policy_feature_frame(
    s07_competence: pd.DataFrame,
    s08_candidates: pd.DataFrame,
    s08_archive: pd.DataFrame,
    s09_sweeps: pd.DataFrame,
    s09_boundaries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one joined policy-level feature table for S10 embedding."""

    required = {"policy_id", "policy_name", "source_kind"}
    for label, frame in (("S07", s07_competence), ("S08", s08_candidates)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} input missing columns: {missing}")

    archive_ids = set(s08_archive["policy_id"].astype(str)) if "policy_id" in s08_archive.columns else set()
    base_columns = sorted(
        {
            "schema",
            "experiment_id",
            "research_step_id",
            "policy_id",
            "policy_name",
            "source_kind",
            "route",
            "requires_cpu_fallback",
            "classic_dsl_seed",
            "qd_candidate",
            "archive_winner",
            "quality_score",
            *COMPETENCE_FEATURE_COLUMNS,
        }
    )

    frames: list[pd.DataFrame] = []
    for source_label, source_frame, priority in (
        ("s07_competence", s07_competence, 0),
        ("s08_qd_candidates", s08_candidates, 1),
    ):
        selected = source_frame.reindex(columns=base_columns).copy()
        selected["policy_id"] = selected["policy_id"].astype(str)
        selected["embedding_input_source"] = source_label
        selected["s10_priority"] = priority
        if "qd_candidate" not in source_frame.columns:
            selected["qd_candidate"] = source_label == "s08_qd_candidates"
        else:
            selected["qd_candidate"] = selected["qd_candidate"].map(_as_bool)
        selected["s08_archive_winner"] = selected["policy_id"].isin(archive_ids)
        if "archive_winner" in source_frame.columns:
            selected["s08_archive_winner"] = selected["s08_archive_winner"] | selected["archive_winner"].map(_as_bool)
        selected["classic_dsl_seed"] = selected["classic_dsl_seed"].map(_as_bool)
        selected["requires_cpu_fallback"] = selected["requires_cpu_fallback"].map(_as_bool)
        selected["quality_score"] = pd.to_numeric(
            selected.apply(lambda row: _first_present(row, ("quality_score", "screen_score")), axis=1),
            errors="coerce",
        )
        frames.append(selected.drop(columns=["archive_winner"], errors="ignore"))

    policies = pd.concat(frames, ignore_index=True, sort=False)
    policies = (
        policies.sort_values(["policy_id", "s10_priority"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("policy_id", keep="first")
        .sort_values(["embedding_input_source", "policy_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    policies["classic_family"] = [
        classic_family(name, flag)
        for name, flag in zip(policies["policy_name"], policies["classic_dsl_seed"], strict=True)
    ]
    policies["classic_landmark"] = policies["classic_family"].astype(str).ne("")

    phase = aggregate_phase_diagnostics(s09_sweeps, s09_boundaries)
    policies = policies.merge(phase, on="policy_id", how="left", validate="one_to_one")
    policies["s09_diagnostic_available"] = policies["s09_run_count"].notna()
    policies["s09_run_count"] = pd.to_numeric(policies["s09_run_count"], errors="coerce")
    return policies


def behavior_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return numeric behavior columns to include in the embedding."""

    candidates: list[str] = []
    for column in COMPETENCE_FEATURE_COLUMNS:
        if column in frame.columns:
            candidates.append(column)
    candidates.extend(column for column in frame.columns if column.startswith("s09_"))
    excluded = set(METADATA_COLUMNS) | {"s09_diagnostic_available"}
    unique: list[str] = []
    for column in candidates:
        if column in excluded or column in unique:
            continue
        numeric = _safe_numeric(frame, column)
        if numeric.notna().sum() >= 2 and numeric.nunique(dropna=True) >= 2:
            unique.append(column)
    return tuple(unique)


def normalize_behavior_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    method: str = "robust",
    block_balance: bool = True,
) -> NormalizedFeatureMatrix:
    """Impute and normalize behavior features into a finite matrix."""

    if method not in {"robust", "zscore", "minmax"}:
        raise ValueError(f"Unsupported normalization method: {method}")
    if not feature_columns:
        raise ValueError("At least one feature column is required")

    raw = pd.DataFrame({column: _safe_numeric(frame, column) for column in feature_columns}, index=frame.index)
    usable = [
        column
        for column in raw.columns
        if raw[column].notna().sum() >= 2 and raw[column].nunique(dropna=True) >= 2
    ]
    if not usable:
        raise ValueError("No usable behavior feature columns remain after filtering")

    raw = raw[usable]
    missing_fraction = raw.isna().mean(axis=1).to_numpy(dtype=float)
    medians = raw.median(axis=0, skipna=True)
    filled = raw.fillna(medians)

    if method == "robust":
        center = raw.median(axis=0, skipna=True)
        q1 = raw.quantile(0.25, axis=0)
        q3 = raw.quantile(0.75, axis=0)
        scale = q3 - q1
        std = raw.std(axis=0, ddof=0, skipna=True)
        scale = scale.mask(scale.abs() < 1e-12, std)
        scale = scale.mask(scale.abs() < 1e-12, 1.0)
    elif method == "zscore":
        center = raw.mean(axis=0, skipna=True)
        scale = raw.std(axis=0, ddof=0, skipna=True).mask(lambda item: item.abs() < 1e-12, 1.0)
    else:
        min_value = raw.min(axis=0, skipna=True)
        max_value = raw.max(axis=0, skipna=True)
        center = min_value
        scale = (max_value - min_value).mask(lambda item: item.abs() < 1e-12, 1.0)

    normalized = (filled - center) / scale
    finite = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    varying = finite.std(axis=0, ddof=0) > 1e-12
    finite = finite.loc[:, varying]
    if finite.shape[1] < 2:
        raise ValueError("Need at least two varying behavior features for embedding")

    block_for_column = {column: ("phase_diagnostics" if column.startswith("s09_") else "competence") for column in normalized.columns}
    block_weights: dict[str, float] = {column: 1.0 for column in normalized.columns}
    if block_balance:
        retained_blocks = pd.Series([block_for_column[column] for column in finite.columns]).value_counts().to_dict()
        for column in finite.columns:
            block_weights[column] = 1.0 / math.sqrt(float(retained_blocks[block_for_column[column]]))
            finite[column] = finite[column] * block_weights[column]

    stats = pd.DataFrame(
        {
            "feature": list(normalized.columns),
            "normalization_method": method,
            "block": [block_for_column[column] for column in normalized.columns],
            "block_weight": [float(block_weights[column]) for column in normalized.columns],
            "observed_count": [int(raw[column].notna().sum()) for column in normalized.columns],
            "missing_count": [int(raw[column].isna().sum()) for column in normalized.columns],
            "imputed_median": [float(medians[column]) for column in normalized.columns],
            "center": [float(center[column]) for column in normalized.columns],
            "scale": [float(scale[column]) for column in normalized.columns],
            "retained": [bool(varying[column]) for column in normalized.columns],
        }
    )
    return NormalizedFeatureMatrix(
        matrix=finite.to_numpy(dtype=float),
        feature_names=tuple(finite.columns),
        imputed_feature_names=tuple(raw.columns),
        missing_fraction=missing_fraction,
        stats=stats,
        method=method,
    )


def compute_embedding(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    seed: int = DEFAULT_EMBEDDING_SEED,
    normalization: str = "robust",
    n_components: int = 3,
) -> EmbeddingResult:
    """Compute a deterministic PCA behavior embedding."""

    normalized = normalize_behavior_features(frame, feature_columns, method=normalization)
    component_count = min(n_components, normalized.matrix.shape[0], normalized.matrix.shape[1])
    if component_count < 2:
        raise ValueError("Need at least two PCA components")
    pca = PCA(n_components=component_count, svd_solver="full", random_state=seed)
    coords = pca.fit_transform(normalized.matrix)
    if component_count < n_components:
        coords = np.column_stack([coords, np.zeros((coords.shape[0], n_components - component_count), dtype=float)])
    ratios = list(float(value) for value in pca.explained_variance_ratio_)
    ratios.extend([0.0] * (n_components - len(ratios)))
    return EmbeddingResult(
        embedding=coords[:, :n_components],
        explained_variance_ratio=tuple(ratios[:n_components]),  # type: ignore[return-value]
        normalized=normalized,
        pca=pca,
    )


def _neighbor_sets(coordinates: np.ndarray, k: int) -> list[set[int]]:
    neighbor_count = max(1, min(k + 1, len(coordinates) - 1))
    model = NearestNeighbors(n_neighbors=neighbor_count)
    indices = model.fit(coordinates).kneighbors(return_distance=False)
    sets: list[set[int]] = []
    for row_index, row in enumerate(indices):
        neighbors = [int(index) for index in row if int(index) != row_index][:k]
        sets.append(set(neighbors))
    return sets


def _mean_jaccard(left: Sequence[set[int]], right: Sequence[set[int]]) -> float:
    scores: list[float] = []
    for a, b in zip(left, right, strict=True):
        union = a | b
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(a & b) / len(union))
    return float(np.mean(scores))


def embedding_stability_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    primary_embedding: np.ndarray,
    *,
    seeds: Sequence[int] = DEFAULT_STABILITY_SEEDS,
    feature_fraction: float = 0.8,
    neighbor_k: int = 10,
    distance_sample_size: int = 600,
) -> pd.DataFrame:
    """Compare feature-subset and normalization-variant embeddings to primary PCA."""

    feature_columns = tuple(feature_columns)
    if len(feature_columns) < 3:
        raise ValueError("At least three feature columns are needed for stability checks")

    primary_neighbors = _neighbor_sets(primary_embedding[:, :2], neighbor_k)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        subset_size = max(2, int(math.ceil(len(feature_columns) * feature_fraction)))
        subset = tuple(sorted(rng.choice(feature_columns, size=subset_size, replace=False).tolist()))
        for normalization in ("robust", "zscore"):
            result = compute_embedding(frame, subset, seed=seed, normalization=normalization, n_components=2)
            coords = result.embedding[:, :2]
            neighbors = _neighbor_sets(coords, neighbor_k)
            jaccard = _mean_jaccard(primary_neighbors, neighbors)
            sample_size = min(distance_sample_size, len(frame))
            sample = np.sort(rng.choice(len(frame), size=sample_size, replace=False))
            primary_dist = pdist(primary_embedding[sample, :2])
            replicate_dist = pdist(coords[sample, :2])
            corr = spearmanr(primary_dist, replicate_dist).correlation
            if corr is None or pd.isna(corr):
                corr = 0.0
            rows.append(
                {
                    "schema": EMBEDDING_SCHEMA,
                    "experiment_id": "E03",
                    "research_step_id": "S10",
                    "seed": int(seed),
                    "normalization": normalization,
                    "feature_fraction": float(feature_fraction),
                    "feature_count": int(len(subset)),
                    "neighbor_k": int(neighbor_k),
                    "mean_neighbor_jaccard": float(jaccard),
                    "sample_distance_spearman": float(corr),
                    "explained_variance_1": float(result.explained_variance_ratio[0]),
                    "explained_variance_2": float(result.explained_variance_ratio[1]),
                }
            )
    return pd.DataFrame(rows)


def attach_embedding_columns(policy_frame: pd.DataFrame, result: EmbeddingResult) -> pd.DataFrame:
    """Return the policy frame with S10 embedding coordinates and diagnostics."""

    out = policy_frame.copy()
    out["schema"] = EMBEDDING_SCHEMA
    out["experiment_id"] = "E03"
    out["research_step_id"] = "S10"
    out["embedding_x"] = result.embedding[:, 0]
    out["embedding_y"] = result.embedding[:, 1]
    out["embedding_z"] = result.embedding[:, 2]
    out["pca_explained_variance_1"] = result.explained_variance_ratio[0]
    out["pca_explained_variance_2"] = result.explained_variance_ratio[1]
    out["pca_explained_variance_3"] = result.explained_variance_ratio[2]
    out["embedding_feature_count"] = len(result.normalized.feature_names)
    out["embedding_missing_fraction"] = result.normalized.missing_fraction
    return out


def landmark_frame(embedding_frame: pd.DataFrame) -> pd.DataFrame:
    """Return classic landmark rows for plotting and handoff tables."""

    columns = [
        "policy_id",
        "policy_name",
        "classic_family",
        "source_kind",
        "embedding_x",
        "embedding_y",
        "embedding_z",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_work_mean",
        "s09_diagnostic_available",
    ]
    available = [column for column in columns if column in embedding_frame.columns]
    landmarks = embedding_frame[embedding_frame["classic_landmark"].map(_as_bool)].copy()
    return landmarks[available].sort_values(["classic_family", "policy_name"], kind="mergesort")


def validation_frame(
    *,
    policy_frame: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    stability: pd.DataFrame,
    landmarks: pd.DataFrame,
    figure_exists: bool,
    unit_success: bool,
) -> pd.DataFrame:
    """Build S10 validation cases."""

    classic_families = set(landmarks.get("classic_family", pd.Series(dtype=str)).astype(str))
    classic_families.discard("")
    finite_embedding = np.isfinite(embedding_frame[["embedding_x", "embedding_y", "embedding_z"]].to_numpy(dtype=float)).all()
    mean_jaccard = float(stability["mean_neighbor_jaccard"].mean()) if not stability.empty else 0.0
    median_spearman = float(stability["sample_distance_spearman"].median()) if not stability.empty else 0.0
    cases = [
        {
            "validation_case": "s07_rows_joined",
            "success": int((policy_frame["embedding_input_source"] == "s07_competence").sum()) >= 2500,
            "expected": "at least 2500 S07 policies",
            "observed": str(int((policy_frame["embedding_input_source"] == "s07_competence").sum())),
            "notes": "S07 competence table is the main policy library.",
        },
        {
            "validation_case": "s08_rows_joined",
            "success": int((policy_frame["embedding_input_source"] == "s08_qd_candidates").sum()) >= 300,
            "expected": "at least 300 S08 QD candidates",
            "observed": str(int((policy_frame["embedding_input_source"] == "s08_qd_candidates").sum())),
            "notes": "S08 candidates add discovered MAP-Elites behavior.",
        },
        {
            "validation_case": "s09_diagnostics_joined",
            "success": int(policy_frame["s09_diagnostic_available"].sum()) >= 10,
            "expected": "at least 10 policies with S09 diagnostics",
            "observed": str(int(policy_frame["s09_diagnostic_available"].sum())),
            "notes": "S09 phase diagnostics are joined by base policy ID.",
        },
        {
            "validation_case": "unique_policy_ids",
            "success": bool(policy_frame["policy_id"].is_unique),
            "expected": "one row per policy ID",
            "observed": str(policy_frame["policy_id"].nunique()),
            "notes": "Duplicate IDs are resolved by preferring S08 candidate rows when present.",
        },
        {
            "validation_case": "feature_matrix_size",
            "success": len(feature_columns) >= 20,
            "expected": "at least 20 behavior feature columns",
            "observed": str(len(feature_columns)),
            "notes": "Features are competence and trajectory diagnostics, not source grammar flags.",
        },
        {
            "validation_case": "finite_embedding",
            "success": bool(finite_embedding),
            "expected": "all embedding coordinates finite",
            "observed": str(bool(finite_embedding)),
            "notes": "Missing metrics are median-imputed before scaling.",
        },
        {
            "validation_case": "embedding_row_count",
            "success": len(embedding_frame) == len(policy_frame),
            "expected": str(len(policy_frame)),
            "observed": str(len(embedding_frame)),
            "notes": "No policy dropped during embedding.",
        },
        {
            "validation_case": "stability_replicates",
            "success": len(stability) >= 8,
            "expected": "at least 8 seeded normalization/subset checks",
            "observed": str(len(stability)),
            "notes": "S10 compares feature-subset and robust/z-score variants.",
        },
        {
            "validation_case": "stable_distance_structure",
            "success": median_spearman >= 0.60,
            "expected": "median sampled distance Spearman >= 0.60",
            "observed": f"{median_spearman:.6g}",
            "notes": "Pairwise-distance ranking should be stable across seeded feature subsets.",
        },
        {
            "validation_case": "stable_neighborhoods",
            "success": mean_jaccard >= 0.18,
            "expected": "mean top-k neighbor Jaccard >= 0.18",
            "observed": f"{mean_jaccard:.6g}",
            "notes": "A modest threshold is used because the library contains many near-tied low-competence policies.",
        },
        {
            "validation_case": "classic_landmarks_highlighted",
            "success": len(classic_families & {"Bubble", "Insertion", "Selection"}) == 3,
            "expected": "Bubble, Insertion, and Selection landmarks present",
            "observed": ",".join(sorted(classic_families)),
            "notes": "Classic DSL landmarks are flagged for plotting and downstream interpretation.",
        },
        {
            "validation_case": "figure_written",
            "success": bool(figure_exists),
            "expected": "non-empty policy_behavior_embedding.png",
            "observed": str(bool(figure_exists)),
            "notes": "Figure highlights classic landmarks and archive winners.",
        },
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_success),
            "expected": "full E03 unit suite passes",
            "observed": str(bool(unit_success)),
            "notes": "The suite includes S10 join, normalization, PCA, and stability diagnostics tests.",
        },
    ]
    return pd.DataFrame(cases)


def embedding_digest(embedding_frame: pd.DataFrame) -> str:
    """Return a stable digest over policy IDs and coordinates."""

    records: list[dict[str, Any]] = []
    for row in embedding_frame[["policy_id", "embedding_x", "embedding_y", "embedding_z"]].sort_values("policy_id").to_dict(orient="records"):
        records.append(
            {
                "policy_id": row["policy_id"],
                "embedding_x": round(float(row["embedding_x"]), 10),
                "embedding_y": round(float(row["embedding_y"]), 10),
                "embedding_z": round(float(row["embedding_z"]), 10),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
