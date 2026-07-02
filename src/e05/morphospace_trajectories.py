"""Morphospace trajectory embedding helpers for E05 S12."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


STEP_ID = "S12"
FEATURE_COLUMNS = (
    "target_error",
    "aggregate_morphospace_error",
    "displacement_fraction",
    "population_fraction",
    "accepted_swaps_per_site",
    "attempted_swaps_per_site",
    "energy_per_site",
    "event_progress",
    "axis_margin",
    "orientation_gap",
)


@dataclass(frozen=True)
class EmbeddingResult:
    """S12 trajectory embedding outputs."""

    embedding_df: pd.DataFrame
    route_metrics_df: pd.DataFrame
    route_summary_df: pd.DataFrame
    stability_df: pd.DataFrame
    source_df: pd.DataFrame
    metric_source_df: pd.DataFrame
    pca_variance_ratio: tuple[float, ...]


def build_morphospace_embedding(artifacts_dir: Path) -> EmbeddingResult:
    """Load S07-S11 inputs, embed trajectory points, and compute route metrics."""

    artifacts_dir = Path(artifacts_dir)
    point_df, source_df, metric_source_df = build_trajectory_point_table(artifacts_dir)
    embedding_df, variance_ratio = embed_points(point_df)
    route_metrics_df, route_summary_df = compute_route_metrics(embedding_df)
    stability_df = pd.DataFrame(
        [
            seed_split_stability(point_df),
            metric_scaling_stability(point_df),
        ]
    )
    return EmbeddingResult(
        embedding_df=embedding_df,
        route_metrics_df=route_metrics_df,
        route_summary_df=route_summary_df,
        stability_df=stability_df,
        source_df=source_df,
        metric_source_df=metric_source_df,
        pca_variance_ratio=variance_ratio,
    )


def build_trajectory_point_table(artifacts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return normalized S07-S11 trajectory points plus source metadata."""

    artifacts_dir = Path(artifacts_dir)
    trace_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []

    specs = (
        ("S07", artifacts_dir / "traces" / "e05_scrambled_embryo_trace_table.parquet", normalize_s07_trace),
        ("S08", artifacts_dir / "traces" / "e05_regeneration_trace_table.parquet", normalize_s08_trace),
        ("S10", artifacts_dir / "results" / "e05_symmetry_breaking_axis_trace.parquet", normalize_s10_trace),
        ("S11", artifacts_dir / "traces" / "e05_gpu_tissue_trace_table.parquet", normalize_s11_trace),
    )
    for source_step, path, normalizer in specs:
        exists = path.exists()
        row_count = 0
        normalized_count = 0
        if exists:
            raw = pd.read_parquet(path)
            row_count = len(raw)
            normalized = normalizer(raw)
            normalized_count = len(normalized)
            trace_frames.append(normalized)
        source_rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": source_step,
                "source_path": str(path),
                "exists": bool(exists),
                "raw_rows": int(row_count),
                "normalized_rows": int(normalized_count),
                "input_type": "downsampled_trace",
                "used_for_embedding": bool(exists and normalized_count > 0),
                "caveat": "",
            }
        )

    s09_endpoint_path = artifacts_dir / "results" / "e05_scaling_tests.parquet"
    s09_metric_path = artifacts_dir / "results" / "e05_scaling_metric_rows.parquet"
    if s09_endpoint_path.exists() and s09_metric_path.exists():
        s09_endpoints = normalize_s09_endpoint_trace(pd.read_parquet(s09_endpoint_path), pd.read_parquet(s09_metric_path))
        trace_frames.append(s09_endpoints)
        source_rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S09",
                "source_path": f"{s09_endpoint_path}; {s09_metric_path}",
                "exists": True,
                "raw_rows": int(len(pd.read_parquet(s09_endpoint_path))),
                "normalized_rows": int(len(s09_endpoints)),
                "input_type": "endpoint_metric_trace",
                "used_for_embedding": True,
                "caveat": "S09 did not write downsampled trajectories, so S12 represents S09 as initial/final S05 metric endpoints only.",
            }
        )
    else:
        source_rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S09",
                "source_path": f"{s09_endpoint_path}; {s09_metric_path}",
                "exists": False,
                "raw_rows": 0,
                "normalized_rows": 0,
                "input_type": "endpoint_metric_trace",
                "used_for_embedding": False,
                "caveat": "S09 endpoint inputs missing.",
            }
        )

    if not trace_frames:
        raise ValueError("no S07-S11 trajectory inputs were available")
    point_df = pd.concat(trace_frames, ignore_index=True, sort=False)
    point_df = _finalize_trace_features(point_df)
    target_df = target_reference_rows(point_df)
    point_df = pd.concat([point_df, target_df], ignore_index=True, sort=False)
    point_df = _finalize_trace_features(point_df)
    metric_source_df = metric_source_rows(artifacts_dir, point_df)
    return point_df, pd.DataFrame(source_rows), metric_source_df


def normalize_s07_trace(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.copy()
    rows["source_research_step_id"] = "S07"
    rows["source_task_type"] = "scrambled_embryo"
    rows["task_id"] = "S07:" + rows["target_id"].astype(str)
    rows["run_uid"] = (
        "S07|"
        + rows["target_id"].astype(str)
        + "|"
        + rows["policy_id"].astype(str)
        + "|seed"
        + rows["simulation_seed"].astype(str)
    )
    rows["perturbation_type"] = ""
    rows["condition_id"] = ""
    rows["scale_label"] = ""
    rows["trace_origin"] = "S07_downsampled_trace"
    rows["endpoint_only"] = False
    rows["population_count"] = np.nan
    rows["displacement_fraction"] = np.nan
    rows["axis_margin"] = 0.0
    rows["orientation_gap"] = 0.0
    return _select_common_columns(rows)


def normalize_s08_trace(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.copy()
    rows["source_research_step_id"] = "S08"
    rows["source_task_type"] = rows["perturbation_type"].astype(str)
    rows["task_id"] = "S08:" + rows["target_id"].astype(str) + ":" + rows["perturbation_type"].astype(str)
    rows["run_uid"] = (
        "S08|"
        + rows["target_id"].astype(str)
        + "|"
        + rows["perturbation_type"].astype(str)
        + "|"
        + rows["policy_id"].astype(str)
        + "|seed"
        + rows["simulation_seed"].astype(str)
        + "|mask"
        + rows["mask_id"].astype(str).map(_short_hash)
    )
    rows["condition_id"] = ""
    rows["scale_label"] = ""
    rows["trace_origin"] = "S08_downsampled_trace"
    rows["endpoint_only"] = False
    rows["displacement_fraction"] = np.nan
    rows["axis_margin"] = 0.0
    rows["orientation_gap"] = 0.0
    return _select_common_columns(rows)


def normalize_s09_endpoint_trace(summary_df: pd.DataFrame, metric_df: pd.DataFrame) -> pd.DataFrame:
    metric_aggregate = (
        metric_df.groupby(["target_id", "task_type", "config_id", "scale_label", "policy_id", "simulation_seed", "benchmark_state_label"], as_index=False)
        .agg(aggregate_from_s05_metrics=("value", "mean"))
    )
    records: list[dict[str, Any]] = []
    for row in summary_df.to_dict(orient="records"):
        for state_role, event_step, target_error, aggregate_error, displacement in (
            (
                "initial",
                0,
                row["initial_target_error"],
                row["initial_aggregate_morphospace_error"],
                row.get("initial_displacement_fraction", np.nan),
            ),
            (
                "final",
                row["event_cap"],
                row["final_target_error"],
                row["final_aggregate_morphospace_error"],
                row.get("final_displacement_fraction", np.nan),
            ),
        ):
            metric_match = metric_aggregate[
                (metric_aggregate["target_id"] == row["target_id"])
                & (metric_aggregate["task_type"] == row["task_type"])
                & (metric_aggregate["config_id"] == row["config_id"])
                & (metric_aggregate["scale_label"] == row["scale_label"])
                & (metric_aggregate["policy_id"] == row["policy_id"])
                & (metric_aggregate["simulation_seed"] == row["simulation_seed"])
                & (metric_aggregate["benchmark_state_label"] == state_role)
            ]
            records.append(
                {
                    "research_step_id": STEP_ID,
                    "source_research_step_id": "S09",
                    "source_task_type": row["task_type"],
                    "target_id": row["target_id"],
                    "target_kind": row["target_kind"],
                    "task_id": "S09:" + str(row["task_type"]) + ":" + str(row["target_id"]),
                    "policy_id": row["policy_id"],
                    "policy_family": row["policy_family"],
                    "simulation_seed": int(row["simulation_seed"]),
                    "run_uid": (
                        "S09|"
                        + str(row["target_id"])
                        + "|"
                        + str(row["task_type"])
                        + "|"
                        + str(row["policy_id"])
                        + "|seed"
                        + str(row["simulation_seed"])
                    ),
                    "event_step": int(event_step),
                    "event_cap": int(row["event_cap"]),
                    "state_role": state_role,
                    "accepted_swaps": 0 if state_role == "initial" else int(row["accepted_swaps"]),
                    "attempted_swaps": 0 if state_role == "initial" else int(row["attempted_swaps"]),
                    "rejected_actions": 0 if state_role == "initial" else int(row["rejected_actions"]),
                    "wait_actions": 0 if state_role == "initial" else int(row["wait_actions"]),
                    "energy_cost": 0.0 if state_role == "initial" else float(row["total_energy_cost"]),
                    "target_error": float(target_error),
                    "aggregate_morphospace_error": float(aggregate_error),
                    "aggregate_from_s05_endpoint_metrics": (
                        float(metric_match["aggregate_from_s05_metrics"].iloc[0]) if not metric_match.empty else np.nan
                    ),
                    "displacement_fraction": float(displacement) if pd.notna(displacement) else np.nan,
                    "population_count": int(row["population_initial"]),
                    "site_count": int(row["site_count"]),
                    "perturbation_type": row.get("perturbation_type", ""),
                    "condition_id": "",
                    "scale_label": row.get("scale_label", ""),
                    "axis_margin": 0.0,
                    "orientation_gap": 0.0,
                    "trace_origin": "S09_endpoint_metrics",
                    "endpoint_only": True,
                }
            )
    return pd.DataFrame(records)


def normalize_s10_trace(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.copy()
    rows["source_research_step_id"] = "S10"
    rows["source_task_type"] = "symmetry_breaking"
    rows["task_id"] = "S10:" + rows["condition_id"].astype(str)
    rows["run_uid"] = (
        "S10|"
        + rows["condition_id"].astype(str)
        + "|"
        + rows["policy_id"].astype(str)
        + "|seed"
        + rows["simulation_seed"].astype(str)
    )
    rows["perturbation_type"] = ""
    rows["scale_label"] = ""
    rows["trace_origin"] = "S10_axis_trace"
    rows["endpoint_only"] = False
    rows["population_count"] = np.nan
    rows["displacement_fraction"] = np.nan
    rows["orientation_gap"] = (rows["vertical_orientation_error"] - rows["horizontal_orientation_error"]).abs()
    return _select_common_columns(rows)


def normalize_s11_trace(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.copy()
    rows["source_research_step_id"] = "S11"
    rows["source_task_type"] = rows["source_task_type"].astype(str)
    rows["policy_family"] = rows["policy_id"].map(
        {
            "s07_local_target_neighbor_descent": "e05_local_target_aware_control",
            "s07_random_adjacent_swap_control": "e05_no_target_random_control",
        }
    ).fillna("")
    rows["perturbation_type"] = rows["source_task_type"].where(rows["source_task_type"].str.contains("patch"), "")
    rows["condition_id"] = rows["task_id"].where(rows["source_task_type"].eq("symmetry_breaking"), "")
    rows["scale_label"] = rows["task_id"].map(lambda value: "large" if str(value).endswith("_large") else ("small" if str(value).endswith("_small") else ""))
    rows["aggregate_morphospace_error"] = np.nan
    rows["population_count"] = np.nan
    rows["axis_margin"] = 0.0
    rows["orientation_gap"] = 0.0
    rows["trace_origin"] = "S11_gpu_tensor_trace"
    rows["endpoint_only"] = False
    return _select_common_columns(rows)


def target_reference_rows(point_df: pd.DataFrame) -> pd.DataFrame:
    """Create explicit zero-error target-state reference rows."""

    keys = [
        "source_research_step_id",
        "source_task_type",
        "task_id",
        "target_id",
        "target_kind",
        "site_count",
    ]
    records = []
    unique = point_df[keys].drop_duplicates().sort_values(keys).to_dict(orient="records")
    for row in unique:
        run_uid = "target_reference|" + str(row["source_research_step_id"]) + "|" + str(row["task_id"]) + "|" + str(row["target_id"])
        records.append(
            {
                "research_step_id": STEP_ID,
                **row,
                "run_uid": run_uid,
                "policy_id": "target_reference",
                "policy_family": "target_reference",
                "simulation_seed": -1,
                "event_step": 0,
                "event_cap": 1,
                "state_role": "target_reference",
                "accepted_swaps": 0,
                "attempted_swaps": 0,
                "rejected_actions": 0,
                "wait_actions": 0,
                "energy_cost": 0.0,
                "target_error": 0.0,
                "aggregate_morphospace_error": 0.0,
                "aggregate_from_s05_endpoint_metrics": 0.0,
                "displacement_fraction": 0.0,
                "population_count": row["site_count"],
                "perturbation_type": "",
                "condition_id": "",
                "scale_label": "",
                "axis_margin": 0.0,
                "orientation_gap": 0.0,
                "trace_origin": "synthetic_target_reference",
                "endpoint_only": False,
                "is_target_reference": True,
            }
        )
    return pd.DataFrame(records)


def embed_points(point_df: pd.DataFrame, feature_weights: Mapping[str, float] | None = None) -> tuple[pd.DataFrame, tuple[float, ...]]:
    """Fit a standardized PCA embedding and return point coordinates."""

    if any(column not in point_df.columns for column in FEATURE_COLUMNS) or "event_progress" not in point_df.columns:
        point_df = _finalize_trace_features(point_df)
    coords, variance_ratio = _fit_transform_pca(point_df, feature_weights=feature_weights)
    output = point_df.copy()
    output["morph_x"] = coords[:, 0]
    output["morph_y"] = coords[:, 1]
    output["morph_z"] = coords[:, 2]
    output["embedding_method"] = "standardized_weighted_pca"
    output["embedding_feature_columns_json"] = pd.Series([list(FEATURE_COLUMNS)] * len(output)).map(lambda value: ",".join(value))
    return output, variance_ratio


def compute_route_metrics(embedding_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-run path curvature and group route diversity metrics."""

    runs = embedding_df[~embedding_df["is_target_reference"]].copy()
    metric_rows: list[dict[str, Any]] = []
    signatures: dict[str, np.ndarray] = {}
    for run_uid, group in runs.groupby("run_uid", sort=False):
        group = group.sort_values(["event_progress", "event_step"])
        coords = group[["morph_x", "morph_y"]].to_numpy(dtype=float)
        target_errors = group["target_error"].to_numpy(dtype=float)
        if len(coords) > 1:
            segment_lengths = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=1))
            path_length = float(segment_lengths.sum())
            direct_distance = float(np.linalg.norm(coords[-1] - coords[0]))
            curvature = float(path_length / max(direct_distance, 1e-12))
            away_steps = int(np.sum(np.diff(target_errors) > 1e-12))
        else:
            path_length = 0.0
            direct_distance = 0.0
            curvature = 0.0
            away_steps = 0
        progress = group["event_progress"].to_numpy(dtype=float)
        signature = _route_signature(progress, coords)
        signatures[str(run_uid)] = signature
        first = group.iloc[0]
        last = group.iloc[-1]
        metric_rows.append(
            {
                "research_step_id": STEP_ID,
                "run_uid": run_uid,
                "source_research_step_id": first["source_research_step_id"],
                "source_task_type": first["source_task_type"],
                "task_id": first["task_id"],
                "target_id": first["target_id"],
                "target_kind": first["target_kind"],
                "policy_id": first["policy_id"],
                "policy_family": first["policy_family"],
                "simulation_seed": int(first["simulation_seed"]),
                "endpoint_only": bool(first["endpoint_only"]),
                "point_count": int(len(group)),
                "initial_target_error": float(first["target_error"]),
                "final_target_error": float(last["target_error"]),
                "target_error_delta": float(last["target_error"] - first["target_error"]),
                "min_target_error": float(np.nanmin(target_errors)),
                "max_target_error": float(np.nanmax(target_errors)),
                "temporary_away_steps": away_steps,
                "temporary_away_fraction": float(away_steps / max(1, len(target_errors) - 1)),
                "max_overshoot_above_initial": float(max(0.0, np.nanmax(target_errors) - target_errors[0])),
                "embedding_path_length": path_length,
                "embedding_direct_distance": direct_distance,
                "path_curvature": curvature,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    diversity_rows: list[dict[str, Any]] = []
    if not metrics.empty:
        metrics["route_diversity_to_group_mean"] = 0.0
        metrics["group_mean_pairwise_route_distance"] = 0.0
        group_cols = ["source_research_step_id", "source_task_type", "task_id", "policy_id"]
        for group_key, group in metrics.groupby(group_cols, sort=False):
            run_ids = list(group["run_uid"])
            matrix = np.vstack([signatures[run_uid] for run_uid in run_ids])
            centroid = matrix.mean(axis=0)
            distances_to_centroid = np.linalg.norm(matrix - centroid, axis=1)
            mean_pairwise = float(pdist(matrix).mean()) if len(matrix) > 1 else 0.0
            metrics.loc[group.index, "route_diversity_to_group_mean"] = distances_to_centroid
            metrics.loc[group.index, "group_mean_pairwise_route_distance"] = mean_pairwise
            diversity_rows.append(
                {
                    "research_step_id": STEP_ID,
                    "source_research_step_id": group_key[0],
                    "source_task_type": group_key[1],
                    "task_id": group_key[2],
                    "policy_id": group_key[3],
                    "runs": int(len(group)),
                    "mean_pairwise_route_distance": mean_pairwise,
                    "mean_route_diversity_to_group_mean": float(distances_to_centroid.mean()),
                    "mean_path_curvature": float(group["path_curvature"].mean()),
                    "mean_temporary_away_fraction": float(group["temporary_away_fraction"].mean()),
                    "endpoint_only_rate": float(group["endpoint_only"].mean()),
                }
            )
    return metrics, pd.DataFrame(diversity_rows)


def seed_split_stability(point_df: pd.DataFrame) -> dict[str, Any]:
    """Fit even/odd seed embeddings and compare pairwise distances."""

    non_target = ~point_df["is_target_reference"]
    even_mask = point_df["is_target_reference"] | (non_target & (point_df["simulation_seed"].astype(int) % 2 == 0))
    odd_mask = point_df["is_target_reference"] | (non_target & (point_df["simulation_seed"].astype(int) % 2 == 1))
    even_coords = _fit_transform_pca(point_df, fit_mask=even_mask)[0][:, :2]
    odd_coords = _fit_transform_pca(point_df, fit_mask=odd_mask)[0][:, :2]
    score = pairwise_distance_stability(even_coords, odd_coords)
    return {
        "research_step_id": STEP_ID,
        "validation_case": "seed_split_embedding_stability",
        "comparison_type": "even_vs_odd_seed_fit",
        "success": bool(score >= 0.70),
        "stability_score": float(score),
        "threshold": 0.70,
        "detail": "Spearman correlation of sampled pairwise distances after fitting PCA on even-seed versus odd-seed trajectory points.",
    }


def metric_scaling_stability(point_df: pd.DataFrame) -> dict[str, Any]:
    """Compare base embedding to an embedding with altered metric-feature weights."""

    base = _fit_transform_pca(point_df)[0][:, :2]
    weights = {column: 1.0 for column in FEATURE_COLUMNS}
    weights["target_error"] = 1.75
    weights["aggregate_morphospace_error"] = 1.75
    weights["displacement_fraction"] = 1.25
    weights["accepted_swaps_per_site"] = 0.75
    weights["attempted_swaps_per_site"] = 0.75
    weights["energy_per_site"] = 0.75
    altered = _fit_transform_pca(point_df, feature_weights=weights)[0][:, :2]
    score = pairwise_distance_stability(base, altered)
    return {
        "research_step_id": STEP_ID,
        "validation_case": "metric_scaling_embedding_stability",
        "comparison_type": "base_vs_weighted_metric_features",
        "success": bool(score >= 0.70),
        "stability_score": float(score),
        "threshold": 0.70,
        "detail": "Spearman correlation of sampled pairwise distances after upweighting S05-derived target and aggregate error features.",
    }


def pairwise_distance_stability(left: np.ndarray, right: np.ndarray, *, sample_size: int = 500) -> float:
    """Return Spearman correlation between sampled pairwise distances."""

    n_rows = int(left.shape[0])
    if n_rows != int(right.shape[0]) or n_rows < 4:
        return 0.0
    idx = np.linspace(0, n_rows - 1, min(sample_size, n_rows), dtype=int)
    left_dist = pdist(left[idx])
    right_dist = pdist(right[idx])
    if np.allclose(left_dist, 0.0) or np.allclose(right_dist, 0.0):
        return 0.0
    score = spearmanr(left_dist, right_dist).correlation
    return float(0.0 if pd.isna(score) else score)


def metric_source_rows(artifacts_dir: Path, point_df: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("S05", artifacts_dir / "results" / "e05_metric_catalog.parquet", "metric_catalog"),
        ("S07", artifacts_dir / "results" / "e05_scrambled_embryo_metric_rows.parquet", "endpoint_s05_metric_rows"),
        ("S08", artifacts_dir / "results" / "e05_regeneration_metric_rows.parquet", "endpoint_s05_metric_rows"),
        ("S09", artifacts_dir / "results" / "e05_scaling_metric_rows.parquet", "endpoint_s05_metric_rows"),
        ("S10", artifacts_dir / "results" / "e05_symmetry_breaking_metric_rows.parquet", "endpoint_s05_metric_rows"),
    )
    rows = []
    for source_step, path, input_type in specs:
        exists = path.exists()
        row_count = 0
        metric_ids: list[str] = []
        if exists:
            df = pd.read_parquet(path)
            row_count = len(df)
            metric_ids = sorted(str(value) for value in df.get("metric_id", pd.Series(dtype=str)).dropna().unique())
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": source_step,
                "source_path": str(path),
                "input_type": input_type,
                "exists": bool(exists),
                "row_count": int(row_count),
                "metric_id_count": int(len(metric_ids)),
                "metric_ids_json": ",".join(metric_ids),
                "used_for_embedding": bool(exists),
            }
        )
    rows.append(
        {
            "research_step_id": STEP_ID,
            "source_research_step_id": "S12",
            "source_path": "normalized trajectory table",
            "input_type": "s05_aggregate_trace_feature",
            "exists": True,
            "row_count": int(point_df["aggregate_morphospace_error"].notna().sum()),
            "metric_id_count": 1,
            "metric_ids_json": "aggregate_morphospace_error",
            "used_for_embedding": True,
        }
    )
    return pd.DataFrame(rows)


def _select_common_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    defaults: dict[str, Any] = {
        "research_step_id": STEP_ID,
        "source_research_step_id": "",
        "source_task_type": "",
        "task_id": "",
        "target_id": "",
        "target_kind": "",
        "policy_id": "",
        "policy_family": "",
        "simulation_seed": -1,
        "run_uid": "",
        "event_step": 0,
        "event_cap": np.nan,
        "state_role": "",
        "accepted_swaps": 0,
        "attempted_swaps": 0,
        "rejected_actions": 0,
        "wait_actions": 0,
        "energy_cost": 0.0,
        "target_error": np.nan,
        "aggregate_morphospace_error": np.nan,
        "aggregate_from_s05_endpoint_metrics": np.nan,
        "displacement_fraction": np.nan,
        "population_count": np.nan,
        "site_count": np.nan,
        "perturbation_type": "",
        "condition_id": "",
        "scale_label": "",
        "axis_margin": 0.0,
        "orientation_gap": 0.0,
        "trace_origin": "",
        "endpoint_only": False,
        "is_target_reference": False,
    }
    for column, default in defaults.items():
        if column not in rows.columns:
            rows[column] = default
    return rows[list(defaults)]


def _finalize_trace_features(point_df: pd.DataFrame) -> pd.DataFrame:
    df = point_df.copy()
    if "event_cap" not in df.columns:
        df["event_cap"] = np.nan
    run_max = df.groupby("run_uid")["event_step"].transform("max")
    df["event_cap"] = df["event_cap"].fillna(run_max)
    zero_cap = df["event_cap"].astype(float).eq(0.0)
    df.loc[zero_cap, "event_cap"] = run_max[zero_cap]
    df["event_cap"] = df["event_cap"].fillna(1).astype(float)
    df["event_progress"] = (df["event_step"].astype(float) / df["event_cap"].replace(0, 1)).clip(0.0, 1.0)
    missing_role = df["state_role"].astype(str).eq("")
    df.loc[missing_role & (df["event_step"].astype(float) <= 0), "state_role"] = "initial"
    df.loc[missing_role & (df["event_step"].astype(float) >= df["event_cap"].astype(float)), "state_role"] = "final"
    df.loc[df["state_role"].astype(str).eq(""), "state_role"] = "intermediate"
    site_count = df["site_count"].copy()
    site_count = site_count.fillna(df.groupby("run_uid")["population_count"].transform("max"))
    site_count = site_count.fillna(df.groupby("target_id")["population_count"].transform("max"))
    site_count = site_count.fillna(1).replace(0, 1)
    df["site_count"] = site_count.astype(float)
    df["population_count"] = df["population_count"].fillna(df["site_count"])
    df["population_fraction"] = (df["population_count"].astype(float) / df["site_count"].replace(0, 1)).clip(0.0, 2.0)
    df["aggregate_morphospace_error"] = df["aggregate_morphospace_error"].fillna(df["target_error"])
    df["displacement_fraction"] = df["displacement_fraction"].fillna(df["target_error"].clip(lower=0.0, upper=1.0))
    df["accepted_swaps_per_site"] = df["accepted_swaps"].astype(float) / df["site_count"].replace(0, 1)
    df["attempted_swaps_per_site"] = df["attempted_swaps"].astype(float) / df["site_count"].replace(0, 1)
    df["energy_per_site"] = df["energy_cost"].astype(float) / df["site_count"].replace(0, 1)
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["is_target_reference"] = df["state_role"].eq("target_reference")
    return df


def _fit_transform_pca(
    point_df: pd.DataFrame,
    *,
    fit_mask: Sequence[bool] | pd.Series | None = None,
    feature_weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    x = point_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    weights = np.array([float((feature_weights or {}).get(column, 1.0)) for column in FEATURE_COLUMNS], dtype=float)
    weights = np.sqrt(np.maximum(weights, 0.0))
    if fit_mask is None:
        fit_mask_array = np.ones(len(point_df), dtype=bool)
    else:
        fit_mask_array = np.asarray(fit_mask, dtype=bool)
    scaler = StandardScaler()
    scaler.fit(x[fit_mask_array])
    x_scaled = scaler.transform(x) * weights
    pca = PCA(n_components=3, random_state=0)
    pca.fit(x_scaled[fit_mask_array])
    coords = pca.transform(x_scaled)
    return coords, tuple(float(value) for value in pca.explained_variance_ratio_)


def _route_signature(progress: np.ndarray, coords: np.ndarray, bins: int = 12) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros(bins * 2, dtype=float)
    order = np.argsort(progress)
    progress = progress[order]
    coords = coords[order]
    unique_progress, unique_indices = np.unique(progress, return_index=True)
    unique_coords = coords[unique_indices]
    grid = np.linspace(0.0, 1.0, bins)
    if len(unique_progress) == 1:
        interpolated = np.repeat(unique_coords[:1], bins, axis=0)
    else:
        x = np.interp(grid, unique_progress, unique_coords[:, 0])
        y = np.interp(grid, unique_progress, unique_coords[:, 1])
        interpolated = np.column_stack([x, y])
    return interpolated.reshape(-1)


def _short_hash(text: str) -> str:
    value = 0
    for char in str(text):
        value = (value * 131 + ord(char)) % 1_000_000_007
    return f"{value:08d}"
