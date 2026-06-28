"""Trajectory morphospace mapping utilities for E05 S12."""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


TRAJECTORY_MAP_SCHEMA_VERSION = "e05_s12_trajectory_morphospace.v1"
TRAJECTORY_FEATURE_SCHEMA_VERSION = "e05_s12_trajectory_features.v1"


@dataclass(frozen=True)
class TraceSourceSpec:
    """Input trace and run-summary artifacts for one upstream E05 step."""

    source_step_id: str
    trace_path: str
    run_path: str
    summary_path: str | None = None
    trace_kind: str = "cpu_morphology"
    run_id_strategy: str = "native_run_id"


def standard_trace_source_specs(artifacts_dir: str | Path = "/artifacts") -> list[TraceSourceSpec]:
    """Return the S07-S11 trace/run artifacts used by S12."""

    root = Path(artifacts_dir)
    step = root / "research_steps"
    return [
        TraceSourceSpec(
            "S07",
            str(step / "S07" / "scrambled_embryo_trace_examples.parquet"),
            str(step / "S07" / "scrambled_embryo_run_results.parquet"),
            str(step / "S07" / "scrambled_embryo_target_policy_summary.parquet"),
            trace_kind="scrambled_recovery",
        ),
        TraceSourceSpec(
            "S08",
            str(step / "S08" / "regeneration_trace_examples.parquet"),
            str(step / "S08" / "regeneration_run_results.parquet"),
            str(step / "S08" / "regeneration_target_perturbation_policy_summary.parquet"),
            trace_kind="regeneration",
        ),
        TraceSourceSpec(
            "S09",
            str(step / "S09" / "scaling_trace_examples.parquet"),
            str(step / "S09" / "scaling_run_results.parquet"),
            str(step / "S09" / "scaling_target_policy_size_summary.parquet"),
            trace_kind="scale_transfer",
        ),
        TraceSourceSpec(
            "S10",
            str(step / "S10" / "symmetry_trace_examples.parquet"),
            str(step / "S10" / "symmetry_breaking_run_results.parquet"),
            str(step / "S10" / "symmetry_task_policy_summary.parquet"),
            trace_kind="symmetry_breaking",
        ),
        TraceSourceSpec(
            "S11",
            str(step / "S11" / "gpu_sweep_trace_examples.parquet"),
            str(step / "S11" / "gpu_sweep_run_results.parquet"),
            str(step / "S11" / "gpu_sweep_summary.parquet"),
            trace_kind="gpu_label_dynamics",
            run_id_strategy="batch_index_seed_order",
        ),
    ]


def available_trace_sources(specs: Sequence[TraceSourceSpec]) -> tuple[list[TraceSourceSpec], pd.DataFrame]:
    """Filter to available sources and return a provenance catalog."""

    available: list[TraceSourceSpec] = []
    rows: list[dict[str, Any]] = []
    for spec in specs:
        trace_path = Path(spec.trace_path)
        run_path = Path(spec.run_path)
        summary_path = Path(spec.summary_path) if spec.summary_path else None
        is_available = trace_path.exists() and run_path.exists()
        if is_available:
            available.append(spec)
        rows.append(
            {
                "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                "research_step_id": "S12",
                "source_step_id": spec.source_step_id,
                "trace_kind": spec.trace_kind,
                "run_id_strategy": spec.run_id_strategy,
                "trace_path": str(trace_path),
                "run_path": str(run_path),
                "summary_path": str(summary_path) if summary_path else None,
                "trace_available": bool(trace_path.exists()),
                "run_summary_available": bool(run_path.exists()),
                "summary_available": bool(summary_path.exists()) if summary_path else False,
                "trace_size_bytes": int(trace_path.stat().st_size) if trace_path.exists() else 0,
                "run_size_bytes": int(run_path.stat().st_size) if run_path.exists() else 0,
            }
        )
    return available, pd.DataFrame(rows)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.map(lambda item: str(item).strip().lower() in {"true", "1", "yes"}).fillna(False)


def _first_available_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column in df.columns:
            out = out.fillna(_numeric_series(df, column))
    return out


def _extract_seed_from_run_id(run_id: Any) -> float:
    match = re.search(r"seed(\d+)", str(run_id))
    return float(match.group(1)) if match else math.nan


def _fill_metadata_from_runs(trace_df: pd.DataFrame, run_df: pd.DataFrame, spec: TraceSourceSpec) -> pd.DataFrame:
    trace = trace_df.copy()
    runs = run_df.copy()
    if "target_id" not in trace.columns and "task_id" in trace.columns:
        trace["target_id"] = trace["task_id"].astype(str)
    if "target_id" not in runs.columns and "task_id" in runs.columns:
        runs["target_id"] = runs["task_id"].astype(str)

    if "run_id" not in trace.columns and spec.run_id_strategy == "batch_index_seed_order":
        sort_cols = [column for column in ["target_id", "policy_id", "seed"] if column in runs.columns]
        run_meta = runs.sort_values(sort_cols).copy()
        run_meta["batch_index"] = run_meta.groupby(["target_id", "policy_id"], sort=False).cumcount()
        merge_cols = ["target_id", "policy_id", "batch_index"]
        extra_cols = [
            column
            for column in run_meta.columns
            if column not in trace.columns and column not in merge_cols
        ]
        trace = trace.merge(run_meta[merge_cols + extra_cols], on=merge_cols, how="left")
    elif "run_id" in trace.columns and "run_id" in runs.columns:
        run_meta = runs.drop_duplicates("run_id").set_index("run_id")
        for column in run_meta.columns:
            if column not in trace.columns:
                trace[column] = trace["run_id"].map(run_meta[column])

    if "run_id" not in trace.columns:
        key_parts = []
        for column in ["source_step_id", "target_id", "policy_id", "batch_index", "seed"]:
            if column == "source_step_id":
                key_parts.append(spec.source_step_id)
            elif column in trace.columns:
                key_parts.append(trace[column].astype(str))
        if len(key_parts) == 1:
            trace["run_id"] = spec.source_step_id + "::trajectory" + trace.groupby(["target_id", "policy_id"], dropna=False).cumcount().astype(str)
        else:
            run_id = key_parts[0]
            for part in key_parts[1:]:
                run_id = run_id + "::" + part
            trace["run_id"] = run_id

    if "seed" not in trace.columns:
        trace["seed"] = trace["run_id"].map(_extract_seed_from_run_id)
    return trace


def harmonize_trace_table(spec: TraceSourceSpec, trace_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    """Convert one upstream trace artifact into a common state-feature table."""

    trace = _fill_metadata_from_runs(trace_df, run_df, spec)
    trace = trace.copy()
    trace["schema_version"] = TRAJECTORY_FEATURE_SCHEMA_VERSION
    trace["research_step_id"] = "S12"
    trace["source_step_id"] = spec.source_step_id
    trace["source_trace_kind"] = spec.trace_kind
    trace["source_trace_path"] = str(spec.trace_path)
    trace["source_run_path"] = str(spec.run_path)

    if "target_id" not in trace.columns and "task_id" in trace.columns:
        trace["target_id"] = trace["task_id"].astype(str)
    for column in ["target_id", "motif", "policy_id", "policy_family", "run_id"]:
        if column not in trace.columns:
            trace[column] = "unknown"
    trace["trajectory_id"] = trace["run_id"].astype(str)
    trace["seed"] = _numeric_series(trace, "seed")
    trace["step"] = _numeric_series(trace, "step").fillna(0.0)
    trace = trace.sort_values(["trajectory_id", "step"], kind="mergesort").reset_index(drop=True)
    trace["snapshot_index"] = trace.groupby("trajectory_id", sort=False).cumcount()
    max_step = trace.groupby("trajectory_id")["step"].transform("max").replace(0.0, np.nan)
    trace["step_fraction"] = (trace["step"] / max_step).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    composite_error = _numeric_series(trace, "composite_error")
    hamming_error = _numeric_series(trace, "hamming_error")
    pattern_error = 1.0 - _numeric_series(trace, "pattern_score")
    label_error = 1.0 - _numeric_series(trace, "label_match_fraction")
    success_error = (~_bool_series(trace, "success")).astype(float)
    error_proxy = composite_error.fillna(hamming_error).fillna(pattern_error).fillna(label_error)
    if error_proxy.isna().any() and "exact_match" in trace.columns:
        error_proxy = error_proxy.fillna((~_bool_series(trace, "exact_match")).astype(float))
    error_proxy = error_proxy.fillna(success_error).clip(lower=0.0)
    trace["error_proxy"] = error_proxy
    trace["score_proxy"] = (1.0 - trace["error_proxy"]).clip(lower=0.0, upper=1.0)

    initial_error = trace.groupby("trajectory_id")["error_proxy"].transform("first")
    denom = initial_error.where(initial_error.abs() > 1e-12, np.nan)
    trace["progress_fraction"] = ((initial_error - trace["error_proxy"]) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    trace["target_energy_feature"] = _first_available_numeric(trace, ["target_energy_normalized", "target_energy"])
    trace["neighborhood_error_feature"] = _numeric_series(trace, "target_neighborhood_error")
    trace["earth_mover_feature"] = _first_available_numeric(trace, ["earth_mover_normalized", "earth_mover_distance"])
    trace["graph_edit_feature"] = _numeric_series(trace, "graph_edit_normalized")
    trace["boundary_error_feature"] = _numeric_series(trace, "boundary_error")
    trace["topology_error_feature"] = _numeric_series(trace, "topology_error")
    trace["shape_moment_error_feature"] = _numeric_series(trace, "shape_moment_error")
    trace["hausdorff_feature"] = _numeric_series(trace, "hausdorff_normalized")
    trace["edge_disagreement_feature"] = _numeric_series(trace, "edge_disagreement")
    trace["axis_strength_feature"] = _numeric_series(trace, "axis_strength")
    trace["oriented_fraction_feature"] = _numeric_series(trace, "oriented_fraction")
    trace["target_axis_alignment_feature"] = _numeric_series(trace, "target_axis_alignment")
    trace["label_diversity_feature"] = _numeric_series(trace, "label_diversity")
    trace["cell_count_feature"] = _numeric_series(trace, "cell_count")
    trace["missing_position_feature"] = _numeric_series(trace, "missing_position_count")
    trace["extra_position_feature"] = _numeric_series(trace, "extra_position_count")

    action_cols = [
        "accepted_swap_count",
        "swap_count",
        "crawl_count",
        "divide_count",
        "die_count",
        "rotate_count",
        "wait_count",
    ]
    for column in action_cols:
        trace[f"{column}_feature"] = _numeric_series(trace, column)
    action_total = pd.Series(0.0, index=trace.index, dtype="float64")
    for column in ["accepted_swap_count_feature", "swap_count_feature", "crawl_count_feature", "divide_count_feature", "die_count_feature", "rotate_count_feature"]:
        action_total = action_total + trace[column].fillna(0.0)
    trace["action_activity_feature"] = action_total

    optional_metadata = [
        "target_id",
        "motif",
        "policy_id",
        "policy_family",
        "task_id",
        "perturbation_id",
        "perturbation_family",
        "size_class",
        "size_key",
        "scale_rule_id",
        "is_heldout_size",
        "information_scope",
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "device",
        "node_count",
        "width",
        "height",
    ]
    for column in optional_metadata:
        if column not in trace.columns:
            trace[column] = None

    keep_cols = [
        "schema_version",
        "research_step_id",
        "source_step_id",
        "source_trace_kind",
        "source_trace_path",
        "source_run_path",
        "trajectory_id",
        "run_id",
        "target_id",
        "task_id",
        "motif",
        "policy_id",
        "policy_family",
        "seed",
        "step",
        "step_fraction",
        "snapshot_index",
        "snapshot_reason",
        "state_hash",
        "perturbation_id",
        "perturbation_family",
        "size_class",
        "size_key",
        "scale_rule_id",
        "is_heldout_size",
        "information_scope",
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "device",
        "node_count",
        "width",
        "height",
        "error_proxy",
        "score_proxy",
        "progress_fraction",
        "target_energy_feature",
        "neighborhood_error_feature",
        "earth_mover_feature",
        "graph_edit_feature",
        "boundary_error_feature",
        "topology_error_feature",
        "shape_moment_error_feature",
        "hausdorff_feature",
        "edge_disagreement_feature",
        "axis_strength_feature",
        "oriented_fraction_feature",
        "target_axis_alignment_feature",
        "label_diversity_feature",
        "cell_count_feature",
        "missing_position_feature",
        "extra_position_feature",
        "accepted_swap_count_feature",
        "swap_count_feature",
        "crawl_count_feature",
        "divide_count_feature",
        "die_count_feature",
        "rotate_count_feature",
        "wait_count_feature",
        "action_activity_feature",
    ]
    for column in keep_cols:
        if column not in trace.columns:
            trace[column] = np.nan
    return trace[keep_cols]


def load_harmonized_traces(specs: Sequence[TraceSourceSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all available S07-S11 traces and return harmonized rows plus catalog."""

    available, catalog = available_trace_sources(specs)
    frames: list[pd.DataFrame] = []
    for spec in available:
        trace_df = pd.read_parquet(spec.trace_path)
        run_df = pd.read_parquet(spec.run_path)
        frames.append(harmonize_trace_table(spec, trace_df, run_df))
    if not frames:
        return pd.DataFrame(), catalog
    return pd.concat(frames, ignore_index=True), catalog


PATH_FEATURE_COLUMNS = [
    "step_fraction",
    "error_proxy",
    "progress_fraction",
    "score_proxy",
    "edge_disagreement_feature",
    "axis_strength_feature",
    "target_energy_feature",
]


EMBEDDING_FEATURE_COLUMNS = [
    "initial_error_proxy",
    "final_error_proxy",
    "best_error_proxy",
    "worst_error_proxy",
    "mean_error_proxy",
    "std_error_proxy",
    "error_reduction",
    "relative_error_reduction_proxy",
    "area_under_error_curve",
    "monotonicity_error_proxy",
    "temporary_worsening_count",
    "path_length_proxy",
    "direct_distance_proxy",
    "path_curvature_proxy",
    "final_score_proxy",
    "score_gain",
    "final_progress_fraction",
    "mean_progress_fraction",
    "final_edge_disagreement",
    "mean_edge_disagreement",
    "final_axis_strength",
    "mean_axis_strength",
    "final_target_energy",
    "mean_target_energy",
    "cell_count_delta",
    "action_activity_total",
    "duration_steps",
    "snapshot_count",
]


def embedding_feature_columns() -> list[str]:
    """Return the documented S12 trajectory-summary feature list."""

    return list(EMBEDDING_FEATURE_COLUMNS)


def _path_feature_matrix(group: pd.DataFrame) -> np.ndarray:
    matrix = group[[column for column in PATH_FEATURE_COLUMNS if column in group.columns]].astype(float)
    if matrix.empty:
        return np.zeros((len(group), 1), dtype=float)
    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    def fill_column(col: pd.Series) -> pd.Series:
        observed = col.dropna()
        median = float(observed.median()) if len(observed) else 0.0
        return col.fillna(median if math.isfinite(median) else 0.0)

    matrix = matrix.apply(fill_column, axis=0)
    return matrix.to_numpy(dtype=float)


def _trajectory_path_stats(group: pd.DataFrame) -> dict[str, float]:
    ordered = group.sort_values(["step", "snapshot_index"], kind="mergesort")
    errors = ordered["error_proxy"].astype(float).to_numpy()
    steps = ordered["step"].astype(float).to_numpy()
    path_matrix = _path_feature_matrix(ordered)
    if len(path_matrix) > 1:
        deltas = np.diff(path_matrix, axis=0)
        path_length = float(np.linalg.norm(deltas, axis=1).sum())
        direct_distance = float(np.linalg.norm(path_matrix[-1] - path_matrix[0]))
    else:
        path_length = 0.0
        direct_distance = 0.0
    curvature = 1.0 if path_length <= 1e-12 and direct_distance <= 1e-12 else path_length / max(direct_distance, 1e-12)
    error_deltas = np.diff(errors) if len(errors) > 1 else np.array([], dtype=float)
    positive_deltas = error_deltas[error_deltas > 0.0]
    if len(steps) > 1:
        area = float(np.trapezoid(errors, steps) / max(float(steps[-1] - steps[0]), 1e-12))
    else:
        area = float(errors[0]) if len(errors) else math.nan
    return {
        "path_length_proxy": path_length,
        "direct_distance_proxy": direct_distance,
        "path_curvature_proxy": curvature,
        "monotonicity_error_proxy": float(positive_deltas.sum()) if len(positive_deltas) else 0.0,
        "temporary_worsening_count": int(len(positive_deltas)),
        "max_temporary_worsening": float(positive_deltas.max()) if len(positive_deltas) else 0.0,
        "area_under_error_curve": area,
    }


def _last_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return math.nan
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if len(values) else math.nan


def _mean_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return math.nan
    values = pd.to_numeric(group[column], errors="coerce")
    return float(values.mean()) if np.isfinite(values.mean()) else math.nan


def summarize_trajectories(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize snapshot-level features into one row per trajectory."""

    rows: list[dict[str, Any]] = []
    metadata_cols = [
        "source_step_id",
        "source_trace_kind",
        "source_trace_path",
        "source_run_path",
        "run_id",
        "target_id",
        "task_id",
        "motif",
        "policy_id",
        "policy_family",
        "seed",
        "perturbation_id",
        "perturbation_family",
        "size_class",
        "size_key",
        "scale_rule_id",
        "is_heldout_size",
        "information_scope",
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "device",
        "node_count",
        "width",
        "height",
    ]
    for trajectory_id, group in feature_df.groupby("trajectory_id", sort=False):
        ordered = group.sort_values(["step", "snapshot_index"], kind="mergesort")
        errors = ordered["error_proxy"].astype(float)
        initial_error = float(errors.iloc[0])
        final_error = float(errors.iloc[-1])
        reduction = initial_error - final_error
        relative = 0.0 if abs(initial_error) <= 1e-12 else reduction / initial_error
        path_stats = _trajectory_path_stats(ordered)
        row: dict[str, Any] = {
            "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
            "research_step_id": "S12",
            "trajectory_id": str(trajectory_id),
            "initial_error_proxy": initial_error,
            "final_error_proxy": final_error,
            "best_error_proxy": float(errors.min()),
            "worst_error_proxy": float(errors.max()),
            "mean_error_proxy": float(errors.mean()),
            "std_error_proxy": float(errors.std(ddof=0)) if len(errors) > 1 else 0.0,
            "error_reduction": reduction,
            "relative_error_reduction_proxy": relative,
            "initial_score_proxy": float(ordered["score_proxy"].iloc[0]),
            "final_score_proxy": float(ordered["score_proxy"].iloc[-1]),
            "score_gain": float(ordered["score_proxy"].iloc[-1] - ordered["score_proxy"].iloc[0]),
            "final_progress_fraction": float(ordered["progress_fraction"].iloc[-1]),
            "mean_progress_fraction": float(ordered["progress_fraction"].mean()),
            "duration_steps": float(ordered["step"].max() - ordered["step"].min()),
            "snapshot_count": int(len(ordered)),
            "state_hash_start": ordered["state_hash"].iloc[0] if "state_hash" in ordered.columns else None,
            "state_hash_final": ordered["state_hash"].iloc[-1] if "state_hash" in ordered.columns else None,
            "final_edge_disagreement": _last_numeric(ordered, "edge_disagreement_feature"),
            "mean_edge_disagreement": _mean_numeric(ordered, "edge_disagreement_feature"),
            "final_axis_strength": _last_numeric(ordered, "axis_strength_feature"),
            "mean_axis_strength": _mean_numeric(ordered, "axis_strength_feature"),
            "final_target_energy": _last_numeric(ordered, "target_energy_feature"),
            "mean_target_energy": _mean_numeric(ordered, "target_energy_feature"),
            "cell_count_delta": _last_numeric(ordered, "cell_count_feature") - _last_numeric(ordered.iloc[[0]], "cell_count_feature"),
            "action_activity_total": _last_numeric(ordered, "action_activity_feature"),
        }
        row.update(path_stats)
        for column in metadata_cols:
            row[column] = ordered[column].iloc[0] if column in ordered.columns else None
        exact_like = final_error <= 1e-9
        if final_error <= 0.05 or exact_like:
            basin = "near_target"
        elif relative > 0.05:
            basin = "improved_partial"
        elif relative < -0.05:
            basin = "diverged"
        else:
            basin = "stalled"
        row["convergence_basin"] = basin
        row["exact_or_near_target"] = bool(exact_like or final_error <= 0.05)
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_embedding_matrix(summary_df: pd.DataFrame, feature_cols: Sequence[str] | None = None) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return a standardized numeric matrix and imputed feature table."""

    cols = list(feature_cols or EMBEDDING_FEATURE_COLUMNS)
    features = summary_df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    medians = pd.Series(
        {
            column: (float(observed.median()) if len(observed) else 0.0)
            for column, observed in ((column, features[column].replace([np.inf, -np.inf], np.nan).dropna()) for column in features.columns)
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(medians)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(features.to_numpy(dtype=float))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, features, cols


def _embedding_method_info(method: str, matrix: np.ndarray, coords: np.ndarray, model: Any | None = None) -> dict[str, Any]:
    distances = pdist(matrix, metric="euclidean")
    coord_distances = pdist(coords, metric="euclidean")
    corr = float(spearmanr(distances, coord_distances).statistic) if len(distances) > 1 else math.nan
    stress = math.nan
    if len(distances) > 1 and np.isfinite(distances).all():
        denom = float(np.square(distances).sum())
        stress = float(np.sqrt(np.square(distances - coord_distances).sum() / max(denom, 1e-12)))
    info = {
        "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
        "research_step_id": "S12",
        "embedding_method": method,
        "trajectory_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "distance_spearman": corr,
        "normalized_stress": stress,
    }
    if method == "pca" and model is not None:
        ratios = getattr(model, "explained_variance_ratio_", np.array([]))
        info["explained_variance_ratio_1"] = float(ratios[0]) if len(ratios) > 0 else math.nan
        info["explained_variance_ratio_2"] = float(ratios[1]) if len(ratios) > 1 else math.nan
        info["explained_variance_ratio_total"] = float(ratios[:2].sum()) if len(ratios) else math.nan
    return info


def fit_embedding_coordinates(matrix: np.ndarray, random_state: int = 12012) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Fit PCA and diffusion-map embeddings for a standardized feature matrix."""

    if matrix.shape[0] < 2:
        raise ValueError("at least two trajectories are required for embedding")
    n_components = 2 if matrix.shape[0] >= 2 else 1
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_coords = pca.fit_transform(matrix)
    if pca_coords.shape[1] == 1:
        pca_coords = np.column_stack([pca_coords[:, 0], np.zeros(matrix.shape[0])])

    spectral = None
    if matrix.shape[0] <= 3:
        n_neighbors = max(1, matrix.shape[0] - 1)
        spectral_coords = pca_coords.copy()
    else:
        n_neighbors = min(30, matrix.shape[0] - 1)
        if matrix.shape[0] >= 7:
            n_neighbors = max(5, n_neighbors)
        spectral = SpectralEmbedding(
            n_components=2,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Graph is not fully connected.*", category=UserWarning)
            spectral_coords = spectral.fit_transform(matrix)
    coords = {"pca": pca_coords, "diffusion_map_knn": spectral_coords}
    info = pd.DataFrame(
        [
            _embedding_method_info("pca", matrix, pca_coords, model=pca),
            _embedding_method_info("diffusion_map_knn", matrix, spectral_coords, model=spectral),
        ]
    )
    info.loc[info["embedding_method"].eq("diffusion_map_knn"), "n_neighbors"] = n_neighbors
    return coords, info


def embedding_rows(summary_df: pd.DataFrame, coords_by_method: dict[str, np.ndarray]) -> pd.DataFrame:
    """Return long-format embedding rows joined to trajectory metadata and path metrics."""

    metadata_cols = [
        "trajectory_id",
        "source_step_id",
        "source_trace_kind",
        "run_id",
        "target_id",
        "motif",
        "policy_id",
        "policy_family",
        "seed",
        "perturbation_id",
        "perturbation_family",
        "size_class",
        "size_key",
        "is_heldout_size",
        "information_scope",
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "device",
        "convergence_basin",
        "exact_or_near_target",
    ]
    metric_cols = [
        "initial_error_proxy",
        "final_error_proxy",
        "relative_error_reduction_proxy",
        "path_curvature_proxy",
        "monotonicity_error_proxy",
        "temporary_worsening_count",
        "area_under_error_curve",
        "duration_steps",
        "snapshot_count",
    ]
    base = summary_df[[column for column in metadata_cols + metric_cols if column in summary_df.columns]].copy()
    rows: list[pd.DataFrame] = []
    for method, coords in coords_by_method.items():
        frame = base.copy()
        frame["schema_version"] = TRAJECTORY_MAP_SCHEMA_VERSION
        frame["research_step_id"] = "S12"
        frame["embedding_method"] = method
        frame["embedding_x"] = coords[:, 0]
        frame["embedding_y"] = coords[:, 1]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _sample_indices(labels: pd.Series, random_state: int, max_items: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = len(labels)
    if n <= max_items:
        return np.arange(n)
    indices: list[int] = []
    per_label = max(1, max_items // max(1, labels.nunique()))
    for _, group_indices in labels.groupby(labels).groups.items():
        group_array = np.array(list(group_indices), dtype=int)
        take = min(len(group_array), per_label)
        indices.extend(rng.choice(group_array, size=take, replace=False).tolist())
    if len(indices) < max_items:
        remaining = np.setdiff1d(np.arange(n), np.array(indices, dtype=int), assume_unique=False)
        extra = rng.choice(remaining, size=min(len(remaining), max_items - len(indices)), replace=False)
        indices.extend(extra.tolist())
    return np.array(sorted(indices[:max_items]), dtype=int)


def embedding_stability_checks(
    summary_df: pd.DataFrame,
    matrix: np.ndarray,
    random_states: Sequence[int] = (12012, 12013, 12014),
    max_items: int = 300,
) -> pd.DataFrame:
    """Check embedding distance stability under source-stratified subsampling."""

    coords_by_method, _ = fit_embedding_coordinates(matrix, random_state=random_states[0])
    rows: list[dict[str, Any]] = []
    for left_method, right_method in [("pca", "diffusion_map_knn")]:
        idx = _sample_indices(summary_df["source_step_id"].astype(str), random_states[0], max_items)
        left_d = pdist(coords_by_method[left_method][idx], metric="euclidean")
        right_d = pdist(coords_by_method[right_method][idx], metric="euclidean")
        corr = float(spearmanr(left_d, right_d).statistic) if len(left_d) > 1 else math.nan
        rows.append(
            {
                "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                "research_step_id": "S12",
                "embedding_method": f"{left_method}_vs_{right_method}",
                "check_type": "cross_method_distance_correlation",
                "sample_count": int(len(idx)),
                "spearman_distance_correlation": corr,
                "passed": bool(np.isfinite(corr)),
                "detail": "Distance-rank agreement between two low-dimensional maps.",
            }
        )

    for method in ["pca", "diffusion_map_knn"]:
        full_coords = coords_by_method[method]
        for seed in random_states:
            idx = _sample_indices(summary_df["source_step_id"].astype(str), seed, max_items)
            sub_matrix = matrix[idx]
            sub_coords = fit_embedding_coordinates(sub_matrix, random_state=seed)[0][method]
            full_d = pdist(full_coords[idx], metric="euclidean")
            sub_d = pdist(sub_coords, metric="euclidean")
            corr = float(spearmanr(full_d, sub_d).statistic) if len(full_d) > 1 else math.nan
            rows.append(
                {
                    "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                    "research_step_id": "S12",
                    "embedding_method": method,
                    "check_type": "source_stratified_subsample_distance_correlation",
                    "random_state": int(seed),
                    "sample_count": int(len(idx)),
                    "spearman_distance_correlation": corr,
                    "passed": bool(np.isfinite(corr)),
                    "detail": "Distance-rank agreement between full-map and refit subsample map.",
                }
            )
    return pd.DataFrame(rows)


def metadata_confounding_checks(
    embedding_df: pd.DataFrame,
    metadata_fields: Sequence[str] = ("source_step_id", "motif", "policy_family", "convergence_basin"),
    random_state: int = 12012,
) -> pd.DataFrame:
    """Estimate how easily embedding coordinates predict non-state metadata."""

    rows: list[dict[str, Any]] = []
    for method, method_df in embedding_df.groupby("embedding_method", sort=True):
        x = method_df[["embedding_x", "embedding_y"]].to_numpy(dtype=float)
        for field in metadata_fields:
            if field not in method_df.columns:
                continue
            labels = method_df[field].fillna("missing").astype(str)
            counts = labels.value_counts()
            if len(counts) < 2 or counts.min() < 2:
                rows.append(
                    {
                        "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                        "research_step_id": "S12",
                        "embedding_method": method,
                        "metadata_field": field,
                        "computed": False,
                        "cv_accuracy": math.nan,
                        "majority_baseline_accuracy": float(counts.max() / len(labels)) if len(labels) else math.nan,
                        "accuracy_lift": math.nan,
                        "class_count": int(len(counts)),
                        "sample_count": int(len(labels)),
                        "confounding_flag": False,
                        "detail": "Skipped because at least one class had fewer than two examples.",
                    }
                )
                continue
            folds = min(5, int(counts.min()))
            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            scores = cross_val_score(model, x, labels.to_numpy(), cv=cv, scoring="accuracy")
            baseline = float(counts.max() / len(labels))
            accuracy = float(np.mean(scores))
            lift = accuracy - baseline
            rows.append(
                {
                    "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                    "research_step_id": "S12",
                    "embedding_method": method,
                    "metadata_field": field,
                    "computed": True,
                    "cv_accuracy": accuracy,
                    "cv_accuracy_std": float(np.std(scores, ddof=0)),
                    "majority_baseline_accuracy": baseline,
                    "accuracy_lift": lift,
                    "class_count": int(len(counts)),
                    "sample_count": int(len(labels)),
                    "confounding_flag": bool(lift > 0.20 and accuracy > 0.60),
                    "detail": "Balanced logistic regression on two embedding coordinates.",
                }
            )
    return pd.DataFrame(rows)


def assign_route_and_failure_clusters(
    summary_df: pd.DataFrame,
    matrix: np.ndarray,
    random_state: int = 12012,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign broad route clusters and failure-mode clusters."""

    out = summary_df.copy()
    n = len(out)
    route_k = max(2, min(8, int(round(math.sqrt(max(n, 2)) / 2))))
    route_model = KMeans(n_clusters=route_k, random_state=random_state, n_init=20)
    out["route_cluster_id"] = [f"route_{label}" for label in route_model.fit_predict(matrix)]

    failure_mask = (~out["exact_or_near_target"].astype(bool)) & (out["final_error_proxy"].astype(float) > 0.05)
    out["failure_cluster_id"] = None
    if int(failure_mask.sum()) >= 4:
        failure_matrix = matrix[failure_mask.to_numpy()]
        failure_k = max(2, min(6, int(round(math.sqrt(int(failure_mask.sum())) / 3))))
        failure_model = KMeans(n_clusters=failure_k, random_state=random_state + 1, n_init=20)
        out.loc[failure_mask, "failure_cluster_id"] = [f"failure_{label}" for label in failure_model.fit_predict(failure_matrix)]

    cluster_rows: list[dict[str, Any]] = []
    for cluster_field in ["route_cluster_id", "failure_cluster_id", "convergence_basin"]:
        grouped = out.dropna(subset=[cluster_field]).groupby(cluster_field, sort=True)
        for cluster_id, group in grouped:
            cluster_rows.append(
                {
                    "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                    "research_step_id": "S12",
                    "cluster_field": cluster_field,
                    "cluster_id": cluster_id,
                    "trajectory_count": int(len(group)),
                    "source_steps": ",".join(sorted(group["source_step_id"].astype(str).unique())),
                    "motifs": ",".join(sorted(group["motif"].astype(str).unique())),
                    "policy_families": ",".join(sorted(group["policy_family"].astype(str).unique())),
                    "mean_final_error_proxy": float(group["final_error_proxy"].astype(float).mean()),
                    "mean_relative_error_reduction": float(group["relative_error_reduction_proxy"].astype(float).mean()),
                    "mean_path_curvature": float(group["path_curvature_proxy"].astype(float).mean()),
                    "mean_temporary_worsening_count": float(group["temporary_worsening_count"].astype(float).mean()),
                }
            )
    return out, pd.DataFrame(cluster_rows)


def route_diversity_summary(summary_df: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    """Summarize path diversity and backtracking by source, motif, and policy family."""

    indexed = summary_df.reset_index(drop=True).copy()
    rows: list[dict[str, Any]] = []
    grouping_sets: list[tuple[str, list[str]]] = [
        ("source_motif_policy_family", ["source_step_id", "motif", "policy_family"]),
        ("motif_policy_family", ["motif", "policy_family"]),
        ("source_policy_family", ["source_step_id", "policy_family"]),
        ("source_step", ["source_step_id"]),
    ]
    for grouping_name, group_cols in grouping_sets:
        for keys, group in indexed.groupby(group_cols, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            idx = group.index.to_numpy(dtype=int)
            if len(idx) > 1:
                distances = pdist(matrix[idx], metric="euclidean")
                mean_distance = float(np.mean(distances))
                max_distance = float(np.max(distances))
            else:
                mean_distance = 0.0
                max_distance = 0.0
            row: dict[str, Any] = {
                "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                "research_step_id": "S12",
                "grouping": grouping_name,
                "trajectory_count": int(len(group)),
                "mean_pairwise_route_distance": mean_distance,
                "max_pairwise_route_distance": max_distance,
                "mean_final_error_proxy": float(group["final_error_proxy"].astype(float).mean()),
                "mean_relative_error_reduction": float(group["relative_error_reduction_proxy"].astype(float).mean()),
                "mean_path_curvature": float(group["path_curvature_proxy"].astype(float).mean()),
                "mean_temporary_worsening_count": float(group["temporary_worsening_count"].astype(float).mean()),
                "near_target_rate": float(group["exact_or_near_target"].astype(bool).mean()),
            }
            for column, value in zip(group_cols, keys, strict=True):
                row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


def validation_rows(
    *,
    source_catalog: pd.DataFrame,
    feature_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    method_info_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    confounding_df: pd.DataFrame,
    route_diversity_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    expected_sources: Iterable[str] = ("S07", "S08", "S09", "S10", "S11"),
) -> pd.DataFrame:
    """Build S12 validation checks."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: Any) -> None:
        rows.append(
            {
                "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
                "research_step_id": "S12",
                "check_id": check_id,
                "success": bool(success),
                "detail": detail,
            }
        )

    expected = set(expected_sources)
    observed = set(feature_df["source_step_id"].astype(str).unique()) if len(feature_df) else set()
    add("source_artifacts_available", expected.issubset(set(source_catalog[source_catalog["trace_available"] & source_catalog["run_summary_available"]]["source_step_id"])), {"expected": sorted(expected), "observed": sorted(observed)})
    add("snapshot_feature_rows_loaded", len(feature_df) > 0, {"featureRows": int(len(feature_df))})
    add("trajectory_identifiers_preserved", bool(feature_df["trajectory_id"].notna().all() and feature_df["source_step_id"].notna().all()), "trajectory_id and source_step_id are populated on all state-feature rows")
    add("all_expected_sources_present", observed == expected, {"expected": sorted(expected), "observed": sorted(observed)})
    add("summary_rows_match_trajectories", len(summary_df) == feature_df["trajectory_id"].nunique(), {"summaryRows": int(len(summary_df)), "trajectoryCount": int(feature_df["trajectory_id"].nunique())})
    feature_finite = np.isfinite(summary_df[EMBEDDING_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)).all()
    add("embedding_features_finite_after_imputation", bool(feature_finite), {"featureColumns": EMBEDDING_FEATURE_COLUMNS})
    add("two_embedding_methods_written", set(embedding_df["embedding_method"].unique()) >= {"pca", "diffusion_map_knn"}, sorted(embedding_df["embedding_method"].unique()))
    add("embedding_rows_preserve_all_trajectories", bool(embedding_df.groupby("embedding_method")["trajectory_id"].nunique().min() == len(summary_df)), {"perMethod": embedding_df.groupby("embedding_method")["trajectory_id"].nunique().to_dict()})
    add("embedding_method_diagnostics_finite", bool(method_info_df["distance_spearman"].map(np.isfinite).all()), method_info_df[["embedding_method", "distance_spearman", "normalized_stress"]].to_dict(orient="records"))
    add("stability_checks_computed", bool(len(stability_df) >= 4 and stability_df["passed"].astype(bool).all()), stability_df[["embedding_method", "check_type", "spearman_distance_correlation", "passed"]].to_dict(orient="records"))
    computed_confound = confounding_df[confounding_df["computed"].astype(bool)] if len(confounding_df) else confounding_df
    add("metadata_confounding_quantified", bool(len(computed_confound) >= 4 and computed_confound["cv_accuracy"].map(np.isfinite).all()), computed_confound[["embedding_method", "metadata_field", "cv_accuracy", "majority_baseline_accuracy", "accuracy_lift", "confounding_flag"]].to_dict(orient="records") if len(computed_confound) else [])
    add("route_diversity_quantified", len(route_diversity_df) > 0, {"rows": int(len(route_diversity_df))})
    add("failure_clusters_or_basins_quantified", len(cluster_df) > 0, {"rows": int(len(cluster_df)), "clusterFields": sorted(cluster_df["cluster_field"].unique()) if len(cluster_df) else []})
    add("path_backtracking_metrics_present", bool((summary_df["temporary_worsening_count"].astype(float) >= 0).all() and (summary_df["monotonicity_error_proxy"].astype(float) >= 0).all()), "monotonicity and temporary-worsening metrics are nonnegative")
    return pd.DataFrame(rows)
