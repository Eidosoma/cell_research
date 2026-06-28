"""Benchmark-suite packaging utilities for E05 S15."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


BENCHMARK_SUITE_SCHEMA_VERSION = "e05_s15_morphology_benchmark_suite.v1"

REQUIRED_SOURCE_ID_COLUMNS = ("source_step_id", "run_id", "trajectory_id")
REQUIRED_INFORMATION_FLAG_COLUMNS = (
    "information_scope",
    "control_class",
    "is_local_only_policy",
    "is_global_information_baseline",
    "uses_target_map",
    "uses_global_gradient",
    "uses_organizer",
)
REQUIRED_DG_COLUMNS = (
    "dg_event_count",
    "productive_dg_event_count",
    "any_dg_event",
    "any_productive_dg_event",
    "normalization_artifact_flag",
    "productive_normalization_artifact_flag",
    "dg_caveat",
    "normalization_warning",
)

REQUIRED_BENCHMARK_IDS = (
    "sort_row",
    "repair_hole",
    "restore_gradient",
    "recover_boundary",
    "regenerate_limb_like_appendage",
    "chimeric_pattern_conflict",
    "scrambled_pattern_recovery",
    "frozen_patch_repair",
    "rotated_graft_repair",
    "symmetry_breaking",
    "higher_dimensional_dg_diagnostics",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    title: str
    benchmark_family: str
    description: str
    primary_goal: str
    selectors: tuple[Mapping[str, tuple[Any, ...]], ...]
    expected_min_rows: int
    required_columns: tuple[str, ...]
    required_artifact_paths: tuple[str, ...]
    metrics: tuple[str, ...]
    caveats: tuple[str, ...]
    validation_notes: tuple[str, ...]
    report_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": BENCHMARK_SUITE_SCHEMA_VERSION,
            "benchmarkId": self.benchmark_id,
            "title": self.title,
            "benchmarkFamily": self.benchmark_family,
            "description": self.description,
            "primaryGoal": self.primary_goal,
            "selectors": [
                {str(key): list(value) for key, value in selector.items()}
                for selector in self.selectors
            ],
            "expectedMinRows": self.expected_min_rows,
            "requiredColumns": list(self.required_columns),
            "requiredArtifactPaths": list(self.required_artifact_paths),
            "metrics": list(self.metrics),
            "caveats": list(self.caveats),
            "validationNotes": list(self.validation_notes),
            "reportTags": list(self.report_tags),
            "dryRunContract": {
                "commandTemplate": "python scripts/e05_s15_benchmark_suite.py --artifacts-dir /artifacts --dry-run-config {config_path}",
                "checks": [
                    "selector_matches_expected_rows",
                    "source_ids_preserved",
                    "local_global_flags_preserved",
                    "dg_caveats_and_normalization_warnings_preserved",
                    "required_artifact_links_resolve",
                ],
            },
        }


def _artifact(artifacts_dir: str | Path, relative: str) -> str:
    return str(Path(artifacts_dir) / relative)


def _selector(**kwargs: Sequence[Any]) -> Mapping[str, tuple[Any, ...]]:
    return {key: tuple(value) for key, value in kwargs.items()}


def standard_benchmark_configs(artifacts_dir: str | Path = "/artifacts") -> list[BenchmarkConfig]:
    """Return the S15 standard morphology benchmark suite configs."""

    common_cols = REQUIRED_SOURCE_ID_COLUMNS + (
        "benchmark_id",
        "target_id",
        "task_id",
        "motif",
        "policy_id",
        "policy_family",
        "final_error",
        "relative_error_reduction",
    )
    flag_cols = REQUIRED_INFORMATION_FLAG_COLUMNS
    dg_cols = REQUIRED_DG_COLUMNS
    common_caveats = (
        "All morphology labels are computational benchmark analogies, not biological anatomy.",
        "Metrics are computational proxies and should be interpreted with source-step caveats.",
    )
    dg_caveat = (
        "DG is measured on saved snapshots using the S14 target-error proxy; normalization warnings must remain attached.",
    )

    return [
        BenchmarkConfig(
            benchmark_id="sort_row",
            title="Sort Row",
            benchmark_family="baseline_preservation",
            description="Row-restricted 1D sorting embedded in 2D plus sorted-row recovery rows.",
            primary_goal="Preserve the E01 sorting semantics inside the E05 framework.",
            selectors=(
                _selector(source_step_id=("S06",)),
                _selector(source_step_id=("S07", "S09"), motif=("sorted_row",)),
            ),
            expected_min_rows=300,
            required_columns=common_cols + flag_cols + dg_cols + ("exact_success",),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S06/embedded_1d_replication_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S06/embedded_1d_replication_report.md"),
                _artifact(artifacts_dir, "results/e05_1d_embedded_replication.parquet"),
            ),
            metrics=("Sortedness", "monotonicity_error", "DG", "Aggregation", "target_error"),
            caveats=common_caveats
            + (
                "S06 Selection preserves E01 same-row ideal-position exchange, which can be non-adjacent.",
                dg_caveat,
            ),
            validation_notes=("S06 exact E01 replay rows must remain available.",),
            report_tags=("baseline", "sort-row", "regression"),
        ),
        BenchmarkConfig(
            benchmark_id="repair_hole",
            title="Repair Hole",
            benchmark_family="regeneration_proxy",
            description="Repair after contiguous center chunk removal.",
            primary_goal="Quantify local pattern repair when target occupancy is damaged.",
            selectors=(_selector(source_step_id=("S08",), perturbation_id=("remove_center_chunk",)),),
            expected_min_rows=80,
            required_columns=common_cols + flag_cols + dg_cols + ("perturbation_id", "perturbation_family"),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S08/regeneration_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S08/perturbation_masks.json"),
                _artifact(artifacts_dir, "figures/e05_s08_regeneration_error_reduction.png"),
            ),
            metrics=("target_error", "cell_count_delta", "boundary_repair", "DG"),
            caveats=common_caveats
            + (
                "Regeneration is proxy-scoped to computational pattern repair.",
                "Division copies local actor identity rather than inferring missing target identity.",
                dg_caveat,
            ),
            validation_notes=("Perturbation mask links must resolve.",),
            report_tags=("regeneration", "hole", "repair"),
        ),
        BenchmarkConfig(
            benchmark_id="restore_gradient",
            title="Restore Gradient",
            benchmark_family="target_recovery",
            description="Gradient and axis-gradient target recovery across CPU, symmetry, and GPU sweeps.",
            primary_goal="Recover or align a target gradient under local-only and explicit global baselines.",
            selectors=(
                _selector(source_step_id=("S07", "S09", "S11"), motif=("gradient",)),
                _selector(source_step_id=("S10",), motif=("axis_gradient",)),
            ),
            expected_min_rows=80,
            required_columns=common_cols + flag_cols + dg_cols + ("uses_global_gradient",),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S07/scrambled_embryo_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S10/symmetry_breaking_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S11/gpu_sweep_run_results.parquet"),
                _artifact(artifacts_dir, "figures/e05_s13_local_global_gaps.png"),
            ),
            metrics=("target_error", "axis_strength", "local_global_gap", "DG"),
            caveats=common_caveats
            + (
                "S10/S11 explicit global-gradient or target-map rows are upper-bound comparators.",
                dg_caveat,
            ),
            validation_notes=("Local/global flags must separate local-only rows from global baselines.",),
            report_tags=("gradient", "local-global", "target-recovery"),
        ),
        BenchmarkConfig(
            benchmark_id="recover_boundary",
            title="Recover Boundary",
            benchmark_family="target_recovery",
            description="Perimeter boundary recovery across scrambled, scaled, and GPU targets.",
            primary_goal="Recover boundary-like computational target constraints.",
            selectors=(_selector(source_step_id=("S07", "S09", "S11"), motif=("boundary",)),),
            expected_min_rows=40,
            required_columns=common_cols + flag_cols + dg_cols,
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S03/target_morphology_library.json"),
                _artifact(artifacts_dir, "research_steps/S05/metric_spec.md"),
                _artifact(artifacts_dir, "figures/e05_s09_scaling_error_reduction.png"),
            ),
            metrics=("boundary_error", "target_energy", "target_error", "DG"),
            caveats=common_caveats
            + (
                "Boundary metrics inherit S05 integer-grid and graph-edit approximation caveats.",
                dg_caveat,
            ),
            validation_notes=("Boundary motif rows and metric documentation must be linked.",),
            report_tags=("boundary", "target-recovery"),
        ),
        BenchmarkConfig(
            benchmark_id="regenerate_limb_like_appendage",
            title="Regenerate Limb-Like Appendage",
            benchmark_family="appendage_proxy",
            description="Asymmetric appendage target rows from symmetry and GPU sweeps.",
            primary_goal="Evaluate abstract appendage-like target restoration and axis selection.",
            selectors=(
                _selector(source_step_id=("S10",), task_id=("asymmetric_appendage_9x9",)),
                _selector(source_step_id=("S11",), motif=("appendage",)),
            ),
            expected_min_rows=40,
            required_columns=common_cols + flag_cols + dg_cols + ("task_id",),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S10/symmetry_breaking_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S11/gpu_sweep_run_results.parquet"),
                _artifact(artifacts_dir, "figures/e05_s10_symmetry_success_heatmap.png"),
            ),
            metrics=("pattern_score", "label_match_fraction", "target_error", "DG"),
            caveats=common_caveats
            + (
                "\"Limb-like\" means abstract asymmetric appendage target only.",
                "Organizer and target-map baselines intentionally carry nonlocal information.",
                dg_caveat,
            ),
            validation_notes=("Appendage rows must preserve explicit baseline flags.",),
            report_tags=("appendage", "symmetry", "gpu"),
        ),
        BenchmarkConfig(
            benchmark_id="chimeric_pattern_conflict",
            title="Chimeric Pattern Conflict",
            benchmark_family="regeneration_proxy",
            description="Foreign-patch and duplicated-region perturbations as abstract conflict tasks.",
            primary_goal="Stress local policies under conflicting or foreign computational patches.",
            selectors=(
                _selector(
                    source_step_id=("S08",),
                    perturbation_id=("insert_foreign_patch_outside", "duplicate_left_region_into_right"),
                ),
            ),
            expected_min_rows=150,
            required_columns=common_cols + flag_cols + dg_cols + ("perturbation_id", "perturbation_family"),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S08/regeneration_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S08/graft_transforms.parquet"),
                _artifact(artifacts_dir, "research_steps/S08/perturbation_masks.json"),
            ),
            metrics=("target_error", "cell_count_delta", "nonconservative_actions", "DG"),
            caveats=common_caveats
            + (
                "Chimeric-pattern conflict is an abstract E05 perturbation label, not a biological chimera claim.",
                "Inserted foreign patches are off-substrate metric stress tests.",
                dg_caveat,
            ),
            validation_notes=("Graft transforms and perturbation masks must resolve.",),
            report_tags=("conflict", "foreign-patch", "regeneration"),
        ),
        BenchmarkConfig(
            benchmark_id="scrambled_pattern_recovery",
            title="Scrambled Pattern Recovery",
            benchmark_family="scrambled_recovery",
            description="Severe 2D disorganization recovery across core S03 target motifs.",
            primary_goal="Recover target patterns from scrambled cell positions.",
            selectors=(_selector(source_step_id=("S07",)),),
            expected_min_rows=150,
            required_columns=common_cols + flag_cols + dg_cols,
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S07/scrambled_embryo_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S07/scrambling_severity.parquet"),
                _artifact(artifacts_dir, "figures/e05_s07_scrambled_recovery_error_reduction.png"),
            ),
            metrics=("target_error", "scramble_severity", "recovery_time", "DG"),
            caveats=common_caveats
            + (
                "\"Embryo\" language is an analogy for scrambled computational tissues.",
                "Scalar rank is a strong visible cue for several S03 targets.",
                dg_caveat,
            ),
            validation_notes=("All S07 source run and trajectory identifiers must be preserved.",),
            report_tags=("scrambled", "recovery", "core-targets"),
        ),
        BenchmarkConfig(
            benchmark_id="frozen_patch_repair",
            title="Frozen Patch Repair",
            benchmark_family="regeneration_proxy",
            description="Repair under immobile center-patch constraints.",
            primary_goal="Quantify repair behavior when a local region cannot move.",
            selectors=(_selector(source_step_id=("S08",), perturbation_id=("freeze_center_patch",)),),
            expected_min_rows=80,
            required_columns=common_cols + flag_cols + dg_cols + ("perturbation_id", "perturbation_family"),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S08/regeneration_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S08/policy_leakage_audit.parquet"),
                _artifact(artifacts_dir, "figures/e05_s08_regeneration_error_reduction.png"),
            ),
            metrics=("target_error", "frozen_violation_count", "target_energy", "DG"),
            caveats=common_caveats
            + (
                "Frozen patches are computational immobility constraints.",
                dg_caveat,
            ),
            validation_notes=("Frozen-patch rows should preserve leakage-audit context.",),
            report_tags=("frozen", "repair", "regeneration"),
        ),
        BenchmarkConfig(
            benchmark_id="rotated_graft_repair",
            title="Rotated Graft Repair",
            benchmark_family="regeneration_proxy",
            description="Repair after a rotated center-graft transform.",
            primary_goal="Recover target morphology after local orientation conflict.",
            selectors=(_selector(source_step_id=("S08",), perturbation_id=("rotate_center_graft",)),),
            expected_min_rows=80,
            required_columns=common_cols + flag_cols + dg_cols + ("perturbation_id", "perturbation_family"),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S08/regeneration_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S08/graft_transforms.parquet"),
                _artifact(artifacts_dir, "figures/e05_s08_cell_count_repair_summary.png"),
            ),
            metrics=("target_error", "graft_transform", "trajectory_curvature", "DG"),
            caveats=common_caveats
            + (
                "Rotated grafts are computational coordinate transforms, not tissue graft experiments.",
                dg_caveat,
            ),
            validation_notes=("Graft-transform provenance must resolve.",),
            report_tags=("rotated-graft", "repair", "regeneration"),
        ),
        BenchmarkConfig(
            benchmark_id="symmetry_breaking",
            title="Symmetry Breaking",
            benchmark_family="symmetry",
            description="D4-symmetric neutral initial states with local and explicit organizer/global baselines.",
            primary_goal="Measure whether policies select target axes or patterns from symmetric starts.",
            selectors=(_selector(source_step_id=("S10",)),),
            expected_min_rows=150,
            required_columns=common_cols + flag_cols + dg_cols + ("task_id", "uses_global_gradient", "uses_organizer"),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S10/symmetry_breaking_run_results.parquet"),
                _artifact(artifacts_dir, "research_steps/S10/symmetric_initial_state_audit.parquet"),
                _artifact(artifacts_dir, "figures/e05_s10_axis_consistency.png"),
            ),
            metrics=("axis_consistency", "pattern_score", "success", "DG"),
            caveats=common_caveats
            + (
                "Local-only policies can break symmetry stochastically but lack a target-axis cue by construction.",
                "Organizer and global-gradient baselines intentionally carry nonlocal information.",
                dg_caveat,
            ),
            validation_notes=("Symmetric initial-state audits and baseline flags must resolve.",),
            report_tags=("symmetry", "axis", "local-global"),
        ),
        BenchmarkConfig(
            benchmark_id="higher_dimensional_dg_diagnostics",
            title="Higher-Dimensional DG Diagnostics",
            benchmark_family="trajectory_diagnostics",
            description="S14 DG event, null-comparison, and normalization-warning diagnostics.",
            primary_goal="Attach higher-dimensional DG measurements and caveats to reusable benchmark records.",
            selectors=(_selector(source_step_id=("S07", "S08", "S09", "S10", "S11")),),
            expected_min_rows=1935,
            required_columns=common_cols + flag_cols + dg_cols + ("total_productive_dg_index",),
            required_artifact_paths=(
                _artifact(artifacts_dir, "research_steps/S14/higher_dimensional_dg_summary.parquet"),
                _artifact(artifacts_dir, "research_steps/S14/dg_normalization_artifact_audit.parquet"),
                _artifact(artifacts_dir, "research_steps/S14/dg_null_comparison_summary.parquet"),
                _artifact(artifacts_dir, "figures/e05_s14_normalization_artifact_audit.png"),
            ),
            metrics=("DG events", "productive DG", "normalization artifacts", "null comparison"),
            caveats=common_caveats
            + (
                "Observed productive DG rates were lower than synthetic and empirical random local-move null rates.",
                "Normalization artifact flags indicate proxy-driven events that require cautious interpretation.",
                dg_caveat,
            ),
            validation_notes=("DG caveat and normalization-warning fields must be nonempty.",),
            report_tags=("DG", "diagnostics", "normalization"),
        ),
    ]


def config_catalog_rows(configs: Sequence[BenchmarkConfig]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for order, config in enumerate(configs, start=1):
        rows.append(
            {
                "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                "research_step_id": "S15",
                "benchmark_order": order,
                "benchmark_id": config.benchmark_id,
                "title": config.title,
                "benchmark_family": config.benchmark_family,
                "description": config.description,
                "primary_goal": config.primary_goal,
                "selector_count": len(config.selectors),
                "expected_min_rows": int(config.expected_min_rows),
                "required_column_count": len(config.required_columns),
                "required_artifact_count": len(config.required_artifact_paths),
                "metrics_json": list(config.metrics),
                "caveats_json": list(config.caveats),
                "validation_notes_json": list(config.validation_notes),
                "report_tags_json": list(config.report_tags),
            }
        )
    return pd.DataFrame(rows)


def mask_for_selectors(df: pd.DataFrame, selectors: Sequence[Mapping[str, Sequence[Any]]]) -> pd.Series:
    if not selectors:
        return pd.Series(True, index=df.index)
    total = pd.Series(False, index=df.index)
    for selector in selectors:
        mask = pd.Series(True, index=df.index)
        for column, values in selector.items():
            if column not in df.columns:
                mask &= False
                continue
            allowed = {"" if value is None else str(value) for value in values}
            series = df[column].fillna("").astype(str)
            mask &= series.isin(allowed)
        total |= mask
    return total


def assign_benchmark_membership(df: pd.DataFrame, configs: Sequence[BenchmarkConfig]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config in configs:
        subset = df.loc[mask_for_selectors(df, config.selectors)].copy()
        subset.insert(0, "benchmark_id", config.benchmark_id)
        subset.insert(1, "benchmark_title", config.title)
        subset.insert(2, "benchmark_family", config.benchmark_family)
        subset["benchmark_schema_version"] = BENCHMARK_SUITE_SCHEMA_VERSION
        frames.append(subset)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def artifact_link_rows(configs: Sequence[BenchmarkConfig]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for path_text in config.required_artifact_paths:
            path = Path(path_text)
            rows.append(
                {
                    "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                    "research_step_id": "S15",
                    "benchmark_id": config.benchmark_id,
                    "artifact_path": str(path),
                    "exists": path.exists(),
                    "is_file": path.is_file(),
                    "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                    "link_validation_success": bool(path.exists() and path.is_file() and path.stat().st_size > 0),
                }
            )
    return pd.DataFrame(rows)


def dry_run_benchmark_config(config: BenchmarkConfig, packaged_df: pd.DataFrame) -> dict[str, Any]:
    rows = packaged_df.loc[packaged_df["benchmark_id"].eq(config.benchmark_id)].copy() if "benchmark_id" in packaged_df.columns else pd.DataFrame()
    missing_columns = [column for column in config.required_columns if column not in rows.columns]
    source_id_missing = False
    if len(rows) and not missing_columns:
        source_id_missing = any(rows[column].isna().any() for column in REQUIRED_SOURCE_ID_COLUMNS if column in rows.columns)
    flag_missing_columns = [column for column in REQUIRED_INFORMATION_FLAG_COLUMNS if column not in rows.columns]
    flags_populated = bool(len(rows) > 0 and not flag_missing_columns and rows[list(REQUIRED_INFORMATION_FLAG_COLUMNS)].notna().all().all())
    dg_missing_columns = [column for column in REQUIRED_DG_COLUMNS if column not in rows.columns]
    dg_populated = bool(len(rows) > 0 and not dg_missing_columns and rows[list(REQUIRED_DG_COLUMNS)].notna().all().all())
    source_count = int(rows["source_step_id"].nunique()) if len(rows) and "source_step_id" in rows.columns else 0
    local_only_rows = int(rows["is_local_only_policy"].astype(bool).sum()) if len(rows) and "is_local_only_policy" in rows.columns else 0
    global_rows = int(rows["is_global_information_baseline"].astype(bool).sum()) if len(rows) and "is_global_information_baseline" in rows.columns else 0
    success = bool(
        len(rows) >= config.expected_min_rows
        and not missing_columns
        and not source_id_missing
        and flags_populated
        and dg_populated
    )
    return {
        "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
        "research_step_id": "S15",
        "benchmark_id": config.benchmark_id,
        "title": config.title,
        "row_count": int(len(rows)),
        "expected_min_rows": int(config.expected_min_rows),
        "source_step_count": source_count,
        "local_only_rows": local_only_rows,
        "global_information_baseline_rows": global_rows,
        "missing_required_columns": ",".join(missing_columns),
        "missing_information_flag_columns": ",".join(flag_missing_columns),
        "missing_dg_columns": ",".join(dg_missing_columns),
        "source_id_missing": bool(source_id_missing),
        "information_flags_populated": flags_populated,
        "dg_caveats_populated": dg_populated,
        "dry_run_success": success,
    }


def dry_run_benchmark_configs(configs: Sequence[BenchmarkConfig], packaged_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([dry_run_benchmark_config(config, packaged_df) for config in configs])


def information_flag_preservation_audit(packaged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark_id, group in packaged_df.groupby("benchmark_id", sort=True):
        missing = [column for column in REQUIRED_INFORMATION_FLAG_COLUMNS if column not in group.columns]
        ids_missing = [column for column in REQUIRED_SOURCE_ID_COLUMNS if column not in group.columns or group[column].isna().any()]
        rows.append(
            {
                "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                "research_step_id": "S15",
                "benchmark_id": benchmark_id,
                "row_count": int(len(group)),
                "source_id_columns_preserved": not ids_missing,
                "missing_source_id_columns": ",".join(ids_missing),
                "information_flag_columns_preserved": not missing,
                "missing_information_flag_columns": ",".join(missing),
                "local_only_rows": int(group["is_local_only_policy"].astype(bool).sum()) if "is_local_only_policy" in group.columns else 0,
                "global_information_baseline_rows": int(group["is_global_information_baseline"].astype(bool).sum()) if "is_global_information_baseline" in group.columns else 0,
                "target_map_rows": int(group["uses_target_map"].astype(bool).sum()) if "uses_target_map" in group.columns else 0,
                "global_gradient_rows": int(group["uses_global_gradient"].astype(bool).sum()) if "uses_global_gradient" in group.columns else 0,
                "organizer_rows": int(group["uses_organizer"].astype(bool).sum()) if "uses_organizer" in group.columns else 0,
                "audit_success": bool(not ids_missing and not missing),
            }
        )
    return pd.DataFrame(rows)


def dg_caveat_preservation_audit(packaged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark_id, group in packaged_df.groupby("benchmark_id", sort=True):
        missing = [column for column in REQUIRED_DG_COLUMNS if column not in group.columns]
        caveats_nonempty = bool("dg_caveat" in group.columns and group["dg_caveat"].fillna("").astype(str).str.len().gt(0).all())
        warnings_nonempty = bool(
            "normalization_warning" in group.columns
            and group["normalization_warning"].fillna("").astype(str).str.len().gt(0).all()
        )
        rows.append(
            {
                "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                "research_step_id": "S15",
                "benchmark_id": benchmark_id,
                "row_count": int(len(group)),
                "missing_dg_columns": ",".join(missing),
                "dg_caveats_nonempty": caveats_nonempty,
                "normalization_warnings_nonempty": warnings_nonempty,
                "normalization_artifact_flag_count": int(group["normalization_artifact_flag"].astype(bool).sum())
                if "normalization_artifact_flag" in group.columns
                else 0,
                "productive_normalization_artifact_flag_count": int(group["productive_normalization_artifact_flag"].astype(bool).sum())
                if "productive_normalization_artifact_flag" in group.columns
                else 0,
                "dg_event_count_sum": int(pd.to_numeric(group["dg_event_count"], errors="coerce").fillna(0).sum())
                if "dg_event_count" in group.columns
                else 0,
                "productive_dg_event_count_sum": int(pd.to_numeric(group["productive_dg_event_count"], errors="coerce").fillna(0).sum())
                if "productive_dg_event_count" in group.columns
                else 0,
                "audit_success": bool(not missing and caveats_nonempty and warnings_nonempty),
            }
        )
    return pd.DataFrame(rows)


def validation_rows(
    configs: Sequence[BenchmarkConfig],
    packaged_df: pd.DataFrame,
    dry_run_df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    info_audit_df: pd.DataFrame,
    dg_audit_df: pd.DataFrame,
    report_bundle_paths: Sequence[str | Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: Any) -> None:
        rows.append(
            {
                "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                "research_step_id": "S15",
                "check_id": check_id,
                "success": bool(success),
                "detail": detail,
            }
        )

    observed_ids = set(config.benchmark_id for config in configs)
    required_ids = set(REQUIRED_BENCHMARK_IDS)
    add("required_benchmark_configs_present", required_ids.issubset(observed_ids), {"required": sorted(required_ids), "observed": sorted(observed_ids)})
    add("every_benchmark_config_dry_runs", bool(len(dry_run_df) == len(configs) and dry_run_df["dry_run_success"].astype(bool).all()), {"rows": int(len(dry_run_df))})
    add("every_benchmark_config_has_rows", bool((dry_run_df["row_count"].astype(int) > 0).all()), {"minRows": int(dry_run_df["row_count"].min()) if len(dry_run_df) else 0})
    add("artifact_links_resolve", bool(len(artifact_df) > 0 and artifact_df["link_validation_success"].astype(bool).all()), {"links": int(len(artifact_df))})
    add("source_ids_preserved", bool(len(info_audit_df) > 0 and info_audit_df["source_id_columns_preserved"].astype(bool).all()), {"rows": int(len(info_audit_df))})
    add("local_global_flags_preserved", bool(len(info_audit_df) > 0 and info_audit_df["information_flag_columns_preserved"].astype(bool).all()), {"rows": int(len(info_audit_df))})
    add("dg_caveats_preserved", bool(len(dg_audit_df) > 0 and dg_audit_df["dg_caveats_nonempty"].astype(bool).all()), {"rows": int(len(dg_audit_df))})
    add("normalization_warnings_preserved", bool(len(dg_audit_df) > 0 and dg_audit_df["normalization_warnings_nonempty"].astype(bool).all()), {"rows": int(len(dg_audit_df))})
    bundle_success = all(Path(path).exists() and Path(path).is_file() and Path(path).stat().st_size > 0 for path in report_bundle_paths)
    add("report_bundle_inputs_written", bundle_success, {"paths": [str(path) for path in report_bundle_paths]})
    required_table_cols = set(REQUIRED_SOURCE_ID_COLUMNS + REQUIRED_INFORMATION_FLAG_COLUMNS + REQUIRED_DG_COLUMNS + ("benchmark_id",))
    add("packaged_result_table_contract", required_table_cols.issubset(set(packaged_df.columns)), {"requiredColumnCount": len(required_table_cols), "rowCount": int(len(packaged_df))})
    return pd.DataFrame(rows)
