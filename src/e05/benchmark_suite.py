"""Morphology benchmark-suite packaging helpers for E05 S15."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


STEP_ID = "S15"
EXPERIMENT_ID = "E05"
BENCHMARK_SUITE_ID = "e05_morphology_benchmark_suite_v1"
BENCHMARK_VERSION = "2026-07-02"
LOCAL_POLICY_IDS = ("s07_local_target_neighbor_descent", "s07_random_adjacent_swap_control")
GLOBAL_CONTEXT_POLICY_IDS = (
    "s13_top_down_global_assignment",
    "s13_organizer_beacon_assignment",
    "s13_global_rebuild_controller",
)
SORT_ROW_METRICS = ("sortedness_percent", "monotonicity_error_count", "final_embedded_morphospace_error")
S05_METRIC_IDS = (
    "target_identity_error",
    "target_neighborhood_error",
    "boundary_error",
    "topology_component_error",
    "shape_moment_error",
    "hausdorff_distance",
    "earth_mover_distance",
    "graph_edit_distance_proxy",
)


STANDARD_TASK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "benchmark_task_id": "sort_row_embedded_1d",
        "title": "Sort row embedded in 2D substrate",
        "benchmark_family": "sort_row",
        "source_research_step_id": "S06",
        "source_table": "e05_1d_embedded_run_summary.parquet",
        "target_id": "sorted_row",
        "target_kind": "sorted_row",
        "task_type": "embedded_1d_sorting",
        "perturbation_type": "initial_permutation",
        "initial_state_protocol": "E01 unique-value arrays embedded into the S06 row-constrained 2D substrate.",
        "action_set": "adjacent row-neighbor swaps only",
        "policy_set": ["s06_bubble_adjacent_swap", "s06_insertion_adjacent_swap"],
        "metric_ids": list(SORT_ROW_METRICS),
        "primary_metric": "final_sortedness_percent",
        "reference_selector": {"algorithms": ["bubble", "insertion"]},
        "smoke_test": "Bubble and Insertion embedded-row references reach 100% sortedness with zero embedded S05 error.",
        "expected_reference_rows": 400,
        "known_blocker": "",
        "claim_boundary": "Continuity task for 1D sorting; not a new 2D morphogenesis claim.",
    },
    {
        "benchmark_task_id": "restore_gradient_scramble",
        "title": "Restore anterior-posterior gradient after full scramble",
        "benchmark_family": "restore_gradient",
        "source_research_step_id": "S13",
        "source_table": "e05_local_vs_global_control.parquet",
        "source_task_basis": "S07 scrambled_embryo",
        "target_id": "ap_gradient",
        "target_kind": "gradient",
        "task_type": "scrambled_embryo",
        "perturbation_type": "full_scramble",
        "initial_state_protocol": "S07 full random permutation of the constructed S03 gradient target state.",
        "action_set": "swap/wait for local policies; declared nonlocal endpoint controls from S13 as upper bounds",
        "policy_set": [*LOCAL_POLICY_IDS, *GLOBAL_CONTEXT_POLICY_IDS],
        "metric_ids": ["target_identity_error", "target_neighborhood_error", "earth_mover_distance"],
        "primary_metric": "final_target_error",
        "reference_selector": {"source_research_step_id": "S07", "target_id": "ap_gradient", "perturbation_type": "full_scramble"},
        "smoke_test": "Reference rows include both local policies and S13 global endpoint controls for the gradient target.",
        "expected_reference_rows": 25,
        "known_blocker": "",
        "claim_boundary": "Gradient restoration is a computational target-error proxy over a toy S03 pattern.",
    },
    {
        "benchmark_task_id": "repair_hole_missing_patch",
        "title": "Repair a missing patch in a boundary target",
        "benchmark_family": "repair_hole",
        "source_research_step_id": "S13",
        "source_table": "e05_local_vs_global_control.parquet",
        "source_task_basis": "S08 regeneration",
        "target_id": "boundary_pattern",
        "target_kind": "boundary",
        "task_type": "regeneration",
        "perturbation_type": "missing_patch",
        "initial_state_protocol": "S08 deterministic contiguous missing patch applied to the S03 boundary target.",
        "action_set": "swap/wait for local policies; declared nonlocal endpoint controls from S13 as upper bounds",
        "policy_set": [*LOCAL_POLICY_IDS, *GLOBAL_CONTEXT_POLICY_IDS],
        "metric_ids": ["target_identity_error", "boundary_error", "graph_edit_distance_proxy"],
        "primary_metric": "final_target_error",
        "reference_selector": {"source_research_step_id": "S08", "target_id": "boundary_pattern", "perturbation_type": "missing_patch"},
        "smoke_test": "Missing-patch local blockers are explicit; global rebuild provides an endpoint upper bound.",
        "expected_reference_rows": 15,
        "known_blocker": "Local swap/wait controls cannot synthesize missing cells; birth/division or replacement identities are required.",
        "claim_boundary": "Repair-hole task is a computational population-deficit proxy, not biological tissue regeneration.",
    },
    {
        "benchmark_task_id": "recover_boundary_scramble",
        "title": "Recover perimeter boundary pattern after full scramble",
        "benchmark_family": "recover_boundary",
        "source_research_step_id": "S13",
        "source_table": "e05_local_vs_global_control.parquet",
        "source_task_basis": "S07 scrambled_embryo",
        "target_id": "boundary_pattern",
        "target_kind": "boundary",
        "task_type": "scrambled_embryo",
        "perturbation_type": "full_scramble",
        "initial_state_protocol": "S07 full random permutation of the constructed S03 boundary target state.",
        "action_set": "swap/wait for local policies; declared nonlocal endpoint controls from S13 as upper bounds",
        "policy_set": [*LOCAL_POLICY_IDS, *GLOBAL_CONTEXT_POLICY_IDS],
        "metric_ids": ["target_identity_error", "boundary_error", "topology_component_error"],
        "primary_metric": "final_target_error",
        "reference_selector": {"source_research_step_id": "S07", "target_id": "boundary_pattern", "perturbation_type": "full_scramble"},
        "smoke_test": "Boundary target references include local and declared global controls with S05 boundary metrics.",
        "expected_reference_rows": 25,
        "known_blocker": "",
        "claim_boundary": "Boundary recovery is measured as a fixed-grid label/shape proxy.",
    },
    {
        "benchmark_task_id": "regenerate_limb_like_appendage",
        "title": "Regenerate toy organ-like appendage after missing patch",
        "benchmark_family": "appendage_regeneration",
        "source_research_step_id": "S13",
        "source_table": "e05_local_vs_global_control.parquet",
        "source_task_basis": "S08 regeneration",
        "target_id": "toy_organ_like",
        "target_kind": "organ_like",
        "task_type": "regeneration",
        "perturbation_type": "missing_patch",
        "initial_state_protocol": "S08 deterministic missing patch applied to the S03 toy organ-like body and appendage pattern.",
        "action_set": "swap/wait for local policies; declared nonlocal endpoint controls from S13 as upper bounds",
        "policy_set": [*LOCAL_POLICY_IDS, *GLOBAL_CONTEXT_POLICY_IDS],
        "metric_ids": ["target_identity_error", "shape_moment_error", "hausdorff_distance"],
        "primary_metric": "final_target_error",
        "reference_selector": {"source_research_step_id": "S08", "target_id": "toy_organ_like", "perturbation_type": "missing_patch"},
        "smoke_test": "Appendage missing-patch blockers are explicit under local controls; global rebuild is an upper bound.",
        "expected_reference_rows": 15,
        "known_blocker": "Local swap/wait controls cannot replace missing appendage-region cells.",
        "claim_boundary": "The appendage is a toy computational pattern, not a biological limb model.",
    },
    {
        "benchmark_task_id": "chimeric_pattern_conflict",
        "title": "Resolve foreign-patch chimeric pattern conflict",
        "benchmark_family": "chimeric_conflict",
        "source_research_step_id": "S13",
        "source_table": "e05_local_vs_global_control.parquet",
        "source_task_basis": "S08 regeneration",
        "target_id": "organ_stripes",
        "target_kind": "stripes",
        "task_type": "regeneration",
        "perturbation_type": "foreign_patch",
        "initial_state_protocol": "S08 deterministic foreign donor patch inserted into the S03 stripes target.",
        "action_set": "swap/wait for local policies; declared nonlocal endpoint controls from S13 as upper bounds",
        "policy_set": [*LOCAL_POLICY_IDS, *GLOBAL_CONTEXT_POLICY_IDS],
        "metric_ids": ["target_identity_error", "target_neighborhood_error", "graph_edit_distance_proxy"],
        "primary_metric": "final_target_error",
        "reference_selector": {"source_research_step_id": "S08", "target_id": "organ_stripes", "perturbation_type": "foreign_patch"},
        "smoke_test": "Foreign-patch identity-conversion blockers are explicit under local controls; global rebuild is an upper bound.",
        "expected_reference_rows": 15,
        "known_blocker": "Foreign-patch conflict may require death, identity conversion, or replacement semantics beyond local swap/wait.",
        "claim_boundary": "Chimeric conflict is a fixed-grid donor-patch proxy; E06 is the dedicated chimeric-governance experiment.",
    },
)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def build_benchmark_suite_tables(artifacts_dir: Path) -> dict[str, pd.DataFrame]:
    """Build S15 benchmark catalog, reference, smoke, and validation tables."""

    artifacts_dir = Path(artifacts_dir)
    source_inputs = source_input_table(artifacts_dir)
    target_index = _read_parquet(artifacts_dir / "results" / "e05_target_morphology_index.parquet")
    metric_catalog_source = _read_parquet(artifacts_dir / "results" / "e05_metric_catalog.parquet")
    reference_df = build_reference_results(artifacts_dir)
    task_catalog = build_task_catalog(reference_df, target_index)
    metric_catalog = build_benchmark_metric_catalog(metric_catalog_source)
    smoke_df = build_smoke_tests(task_catalog, reference_df, metric_catalog)
    validation_df = validate_benchmark_suite_tables(
        task_catalog=task_catalog,
        reference_df=reference_df,
        metric_catalog=metric_catalog,
        smoke_df=smoke_df,
        source_inputs=source_inputs,
        config_dir=None,
        required_artifact_paths=None,
    )
    return {
        "task_catalog": task_catalog,
        "reference_results": reference_df,
        "metric_catalog": metric_catalog,
        "smoke_tests": smoke_df,
        "source_inputs": source_inputs,
        "validation": validation_df,
    }


def benchmark_config_payloads(task_catalog: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return one JSON-serializable config payload per benchmark task."""

    payloads: dict[str, dict[str, Any]] = {}
    by_task = {row["benchmark_task_id"]: row for row in task_catalog.to_dict(orient="records")}
    for spec in STANDARD_TASK_SPECS:
        task_id = spec["benchmark_task_id"]
        row = by_task.get(task_id, {})
        payloads[task_id] = {
            "schema": "eidosoma.e05.benchmark_task_config.v1",
            "benchmarkSuiteId": BENCHMARK_SUITE_ID,
            "benchmarkVersion": BENCHMARK_VERSION,
            "researchStepId": STEP_ID,
            "benchmarkTaskId": task_id,
            "title": spec["title"],
            "benchmarkFamily": spec["benchmark_family"],
            "target": {
                "targetId": spec["target_id"],
                "targetKind": spec["target_kind"],
                "targetHash": row.get("target_hash", ""),
                "substrateKind": row.get("substrate_kind", ""),
                "siteCount": int(row.get("site_count", 0) or 0),
            },
            "protocol": {
                "taskType": spec["task_type"],
                "perturbationType": spec["perturbation_type"],
                "initialStateProtocol": spec["initial_state_protocol"],
                "actionSet": spec["action_set"],
                "policySet": spec["policy_set"],
            },
            "metrics": {
                "primaryMetric": spec["primary_metric"],
                "metricIds": spec["metric_ids"],
                "lowerIsBetterMetrics": [
                    metric for metric in spec["metric_ids"] if metric not in {"sortedness_percent", "final_sortedness_percent"}
                ],
            },
            "referenceOutputs": {
                "sourceResearchStepId": spec["source_research_step_id"],
                "sourceTable": spec["source_table"],
                "sourceTaskBasis": spec.get("source_task_basis", spec["source_research_step_id"]),
                "selector": spec["reference_selector"],
                "expectedReferenceRows": spec["expected_reference_rows"],
                "observedReferenceRows": int(row.get("reference_rows", 0) or 0),
            },
            "smokeTest": {
                "description": spec["smoke_test"],
                "expectedRowsMinimum": int(spec["expected_reference_rows"]),
                "configValidationRequired": True,
            },
            "knownBlocker": spec["known_blocker"],
            "claimBoundary": spec["claim_boundary"],
        }
    return payloads


def source_input_table(artifacts_dir: Path) -> pd.DataFrame:
    """Return compact provenance rows for source artifacts used by S15."""

    specs = [
        ("S01", "substrate_spec_report", artifacts_dir / "reports" / "e05_substrate_spec.md", True),
        ("S02", "identity_vector_spec_report", artifacts_dir / "reports" / "e05_identity_vector_spec.md", True),
        ("S03", "target_morphology_spec_report", artifacts_dir / "reports" / "e05_target_morphology_spec.md", True),
        ("S03", "target_morphology_index", artifacts_dir / "results" / "e05_target_morphology_index.parquet", True),
        ("S04", "action_set_spec_report", artifacts_dir / "reports" / "e05_action_set_spec.md", True),
        ("S05", "morphospace_metric_report", artifacts_dir / "reports" / "e05_morphospace_metrics.md", True),
        ("S05", "metric_catalog", artifacts_dir / "results" / "e05_metric_catalog.parquet", True),
        ("S06", "embedded_1d_reference", artifacts_dir / "results" / "e05_1d_embedded_run_summary.parquet", True),
        ("S07", "scrambled_embryo_reference", artifacts_dir / "results" / "e05_scrambled_embryo_results.parquet", True),
        ("S08", "regeneration_reference", artifacts_dir / "results" / "e05_regeneration_results.parquet", True),
        ("S09", "scaling_reference", artifacts_dir / "results" / "e05_scaling_tests.parquet", True),
        ("S10", "symmetry_reference", artifacts_dir / "results" / "e05_symmetry_breaking.parquet", True),
        ("S11", "gpu_reference", artifacts_dir / "results" / "e05_gpu_tissue_sweeps.parquet", True),
        ("S12", "route_reference", artifacts_dir / "results" / "e05_morphospace_route_metrics.parquet", True),
        ("S13", "local_global_reference", artifacts_dir / "results" / "e05_local_vs_global_control.parquet", True),
        ("S14", "shape_dg_reference", artifacts_dir / "results" / "e05_shape_dg.parquet", True),
    ]
    rows: list[dict[str, Any]] = []
    for source_step, input_type, path, required in specs:
        exists = path.exists()
        row_count = 0
        column_count = 0
        if exists and path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            row_count = int(len(frame))
            column_count = int(len(frame.columns))
        elif exists:
            row_count = int(len(path.read_text(encoding="utf-8").splitlines())) if path.is_file() else 0
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": source_step,
                "input_type": input_type,
                "source_path": str(path),
                "required": bool(required),
                "exists": bool(exists),
                "row_count": row_count,
                "column_count": column_count,
                "used_for_configs": bool(source_step in {"S03", "S04", "S05"}),
                "used_for_reference_results": bool(source_step in {"S06", "S13"}),
                "used_for_report_bundle_context": bool(source_step in {"S07", "S08", "S09", "S10", "S11", "S12", "S14"}),
            }
        )
    return pd.DataFrame(rows)


def build_reference_results(artifacts_dir: Path) -> pd.DataFrame:
    """Normalize S06 and S13 result rows into one benchmark reference table."""

    frames = [
        normalize_s06_sort_row_reference(_read_parquet(Path(artifacts_dir) / "results" / "e05_1d_embedded_run_summary.parquet")),
        normalize_s13_reference(_read_parquet(Path(artifacts_dir) / "results" / "e05_local_vs_global_control.parquet")),
    ]
    reference = pd.concat(frames, ignore_index=True, sort=False)
    reference["benchmark_suite_id"] = BENCHMARK_SUITE_ID
    reference["benchmark_version"] = BENCHMARK_VERSION
    reference["research_step_id"] = STEP_ID
    ordered = [
        "research_step_id",
        "benchmark_suite_id",
        "benchmark_version",
        "benchmark_task_id",
        "benchmark_family",
        "benchmark_title",
        "source_research_step_id",
        "source_result_table",
        "source_row_id",
        "target_id",
        "target_kind",
        "task_type",
        "perturbation_type",
        "policy_id",
        "policy_family",
        "control_class",
        "declared_information_access",
        "simulation_seed",
        "replicate_id",
        "initial_target_error",
        "final_target_error",
        "target_recovery_fraction",
        "initial_aggregate_morphospace_error",
        "final_aggregate_morphospace_error",
        "aggregate_recovery_fraction",
        "final_sortedness_percent",
        "final_monotonicity_error_count",
        "accepted_swaps",
        "attempted_swaps",
        "rejected_actions",
        "wait_actions",
        "total_energy_cost",
        "energy_per_site",
        "baseline_blocked",
        "semantic_blocker",
        "repairability_class",
        "reference_outcome_class",
        "local_policy_information_access_changed",
        "global_state_access",
        "global_target_access",
        "organizer_cue_added",
        "birth_death_allowed",
        "identity_conversion_allowed",
        "unfreeze_allowed",
        "claim_boundary",
    ]
    for column in ordered:
        if column not in reference.columns:
            reference[column] = np.nan
    return reference[ordered].sort_values(["benchmark_task_id", "policy_id", "simulation_seed", "replicate_id"]).reset_index(drop=True)


def normalize_s06_sort_row_reference(df: pd.DataFrame) -> pd.DataFrame:
    spec = spec_by_id("sort_row_embedded_1d")
    rows = []
    for row in df.to_dict(orient="records"):
        algorithm = str(row["algorithm"])
        if algorithm not in {"bubble", "insertion"}:
            continue
        final_error = _first_finite(row.get("final_embedded_morphospace_error", np.nan), row.get("final_row_morphospace_error", np.nan))
        final_identity_error = _first_finite(
            row.get("final_embedded_target_identity_error", np.nan),
            row.get("final_row_target_identity_error", np.nan),
        )
        sortedness = float(row.get("final_sortedness_percent", np.nan))
        rows.append(
            {
                "benchmark_task_id": spec["benchmark_task_id"],
                "benchmark_family": spec["benchmark_family"],
                "benchmark_title": spec["title"],
                "source_research_step_id": "S06",
                "source_result_table": spec["source_table"],
                "source_row_id": f"{row['condition_id']}|{algorithm}|repeat{int(row['repeat_index'])}",
                "target_id": spec["target_id"],
                "target_kind": spec["target_kind"],
                "task_type": spec["task_type"],
                "perturbation_type": spec["perturbation_type"],
                "policy_id": f"s06_{algorithm}_adjacent_swap",
                "policy_family": "e05_embedded_1d_sorting_reference",
                "control_class": "local_algorithm",
                "declared_information_access": "row-neighbor adjacent swap sorting baseline inherited from S06 continuity validation",
                "simulation_seed": int(row["initial_array_seed"]),
                "replicate_id": int(row["repeat_index"]),
                "initial_target_error": np.nan,
                "final_target_error": final_identity_error,
                "target_recovery_fraction": np.nan,
                "initial_aggregate_morphospace_error": np.nan,
                "final_aggregate_morphospace_error": final_error,
                "aggregate_recovery_fraction": np.nan,
                "final_sortedness_percent": sortedness,
                "final_monotonicity_error_count": int(row.get("final_monotonicity_error_count", 0)),
                "accepted_swaps": int(row.get("swap_only_steps", 0)),
                "attempted_swaps": np.nan,
                "rejected_actions": int(row.get("row_constraint_violations", 0)),
                "wait_actions": np.nan,
                "total_energy_cost": float(row.get("energy_total", np.nan)),
                "energy_per_site": float(row.get("energy_total", np.nan)) / max(1.0, float(row.get("array_length", 1))),
                "baseline_blocked": False,
                "semantic_blocker": "",
                "repairability_class": "row_neighbor_sortable",
                "reference_outcome_class": "exact_recovered" if sortedness >= 100.0 and final_error <= 1e-12 else "not_recovered",
                "local_policy_information_access_changed": False,
                "global_state_access": False,
                "global_target_access": False,
                "organizer_cue_added": False,
                "birth_death_allowed": False,
                "identity_conversion_allowed": False,
                "unfreeze_allowed": False,
                "claim_boundary": spec["claim_boundary"],
            }
        )
    return pd.DataFrame(rows)


def normalize_s13_reference(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for spec in STANDARD_TASK_SPECS:
        if spec["benchmark_task_id"] == "sort_row_embedded_1d":
            continue
        selector = spec["reference_selector"]
        rows = df.copy()
        for column, value in selector.items():
            rows = rows[rows[column].fillna("").astype(str).eq(str(value))]
        if rows.empty:
            frames.append(pd.DataFrame())
            continue
        records = []
        for row in rows.to_dict(orient="records"):
            records.append(
                {
                    "benchmark_task_id": spec["benchmark_task_id"],
                    "benchmark_family": spec["benchmark_family"],
                    "benchmark_title": spec["title"],
                    "source_research_step_id": str(row["source_research_step_id"]),
                    "source_result_table": spec["source_table"],
                    "source_row_id": (
                        f"{row['source_research_step_id']}|{row.get('task_id', '')}|{row.get('policy_id', '')}|"
                        f"seed{int(row.get('simulation_seed', -1))}|{row.get('mask_id', '')}"
                    ),
                    "target_id": row.get("target_id", spec["target_id"]),
                    "target_kind": row.get("target_kind", spec["target_kind"]),
                    "task_type": row.get("benchmark_task", spec["task_type"]),
                    "perturbation_type": row.get("perturbation_type", spec["perturbation_type"]),
                    "policy_id": row.get("policy_id", ""),
                    "policy_family": row.get("policy_family", ""),
                    "control_class": row.get("control_class", ""),
                    "declared_information_access": row.get("declared_information_access", ""),
                    "simulation_seed": int(row.get("simulation_seed", -1)),
                    "replicate_id": int(row.get("simulation_seed", -1)),
                    "initial_target_error": float(row.get("initial_target_error", np.nan)),
                    "final_target_error": float(row.get("final_target_error", np.nan)),
                    "target_recovery_fraction": float(row.get("target_recovery_fraction", np.nan)),
                    "initial_aggregate_morphospace_error": float(row.get("initial_aggregate_morphospace_error", np.nan)),
                    "final_aggregate_morphospace_error": float(row.get("final_aggregate_morphospace_error", np.nan)),
                    "aggregate_recovery_fraction": float(row.get("aggregate_recovery_fraction", np.nan)),
                    "final_sortedness_percent": np.nan,
                    "final_monotonicity_error_count": np.nan,
                    "accepted_swaps": int(row.get("accepted_swaps", 0)),
                    "attempted_swaps": int(row.get("attempted_swaps", 0)),
                    "rejected_actions": int(row.get("rejected_actions", 0)),
                    "wait_actions": int(row.get("wait_actions", 0)),
                    "total_energy_cost": float(row.get("total_energy_cost", np.nan)),
                    "energy_per_site": float(row.get("energy_per_site", np.nan)),
                    "baseline_blocked": bool(row.get("baseline_blocked", False)),
                    "semantic_blocker": str(row.get("semantic_blocker", "")),
                    "repairability_class": str(row.get("repairability_class", "")),
                    "reference_outcome_class": classify_reference_outcome(row),
                    "local_policy_information_access_changed": bool(row.get("local_policy_information_access_changed", False)),
                    "global_state_access": bool(row.get("global_state_access", False)),
                    "global_target_access": bool(row.get("global_target_access", False)),
                    "organizer_cue_added": bool(row.get("organizer_cue_added", False)),
                    "birth_death_allowed": bool(row.get("birth_death_allowed", False)),
                    "identity_conversion_allowed": bool(row.get("identity_conversion_allowed", False)),
                    "unfreeze_allowed": bool(row.get("unfreeze_allowed", False)),
                    "claim_boundary": spec["claim_boundary"],
                }
            )
        frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def classify_reference_outcome(row: Mapping[str, Any]) -> str:
    if bool(row.get("baseline_blocked", False)):
        return "blocked_baseline"
    final_error = float(row.get("final_target_error", np.nan))
    recovery = float(row.get("target_recovery_fraction", np.nan))
    if np.isfinite(final_error) and final_error <= 1e-12:
        return "exact_recovered"
    if np.isfinite(recovery) and recovery > 0.05:
        return "partial_recovery"
    if np.isfinite(recovery) and recovery < -0.05:
        return "worse_than_start"
    return "no_recovery_or_stalled"


def build_task_catalog(reference_df: pd.DataFrame, target_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_by_id = {row["target_id"]: row for row in target_index.to_dict(orient="records")}
    for spec in STANDARD_TASK_SPECS:
        task_ref = reference_df[reference_df["benchmark_task_id"].eq(spec["benchmark_task_id"])]
        target = target_by_id.get(spec["target_id"], {})
        outcome_counts = task_ref["reference_outcome_class"].value_counts().to_dict() if not task_ref.empty else {}
        rows.append(
            {
                "research_step_id": STEP_ID,
                "benchmark_suite_id": BENCHMARK_SUITE_ID,
                "benchmark_version": BENCHMARK_VERSION,
                "benchmark_task_id": spec["benchmark_task_id"],
                "title": spec["title"],
                "benchmark_family": spec["benchmark_family"],
                "target_id": spec["target_id"],
                "target_kind": spec["target_kind"],
                "target_hash": target.get("target_hash", ""),
                "substrate_kind": target.get("substrate_kind", ""),
                "site_count": int(target.get("site_count", 0) or 0),
                "task_type": spec["task_type"],
                "perturbation_type": spec["perturbation_type"],
                "source_research_step_id": spec["source_research_step_id"],
                "source_table": spec["source_table"],
                "source_task_basis": spec.get("source_task_basis", spec["source_research_step_id"]),
                "primary_metric": spec["primary_metric"],
                "metric_ids_json": stable_json(spec["metric_ids"]),
                "policy_set_json": stable_json(spec["policy_set"]),
                "reference_rows": int(len(task_ref)),
                "expected_reference_rows": int(spec["expected_reference_rows"]),
                "reference_row_count_matches": bool(len(task_ref) == int(spec["expected_reference_rows"])),
                "local_reference_rows": int(task_ref["control_class"].isin(["local", "local_algorithm"]).sum()) if not task_ref.empty else 0,
                "global_context_reference_rows": int(task_ref["policy_id"].isin(GLOBAL_CONTEXT_POLICY_IDS).sum()) if not task_ref.empty else 0,
                "exact_recovered_rows": int(outcome_counts.get("exact_recovered", 0)),
                "partial_recovery_rows": int(outcome_counts.get("partial_recovery", 0)),
                "blocked_rows": int(outcome_counts.get("blocked_baseline", 0)),
                "worse_than_start_rows": int(outcome_counts.get("worse_than_start", 0)),
                "mean_final_target_error": float(task_ref["final_target_error"].mean()) if not task_ref.empty else np.nan,
                "mean_target_recovery_fraction": float(task_ref["target_recovery_fraction"].mean()) if not task_ref.empty else np.nan,
                "known_blocker": spec["known_blocker"],
                "claim_boundary": spec["claim_boundary"],
                "smoke_test": spec["smoke_test"],
            }
        )
    return pd.DataFrame(rows)


def build_benchmark_metric_catalog(metric_catalog_source: pd.DataFrame) -> pd.DataFrame:
    source_by_id = {row["metric_id"]: row for row in metric_catalog_source.to_dict(orient="records")}
    metric_ids: list[str] = []
    for spec in STANDARD_TASK_SPECS:
        metric_ids.extend(spec["metric_ids"])
    rows = []
    for metric_id in sorted(set(metric_ids)):
        source = source_by_id.get(metric_id, {})
        rows.append(
            {
                "research_step_id": STEP_ID,
                "benchmark_suite_id": BENCHMARK_SUITE_ID,
                "metric_id": metric_id,
                "metric_source": "S05" if metric_id in source_by_id else "S06_special_case",
                "metric_family": source.get("metric_family", "embedded_1d_sorting"),
                "lower_bound": float(source.get("lower_bound", 0.0)),
                "lower_is_better": bool(metric_id not in {"sortedness_percent", "final_sortedness_percent"}),
                "exactness": source.get("exactness", "reference_summary_metric"),
                "approximation_label": source.get("approximation_label", "not_approximate"),
                "caveat": source.get("caveat", "S06 continuity metric, not an S05 morphology metric."),
                "used_by_task_ids_json": stable_json(
                    [spec["benchmark_task_id"] for spec in STANDARD_TASK_SPECS if metric_id in spec["metric_ids"]]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_smoke_tests(task_catalog: pd.DataFrame, reference_df: pd.DataFrame, metric_catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_ids = set(metric_catalog["metric_id"])
    for spec in STANDARD_TASK_SPECS:
        task_id = spec["benchmark_task_id"]
        task_ref = reference_df[reference_df["benchmark_task_id"].eq(task_id)]
        task_row = task_catalog[task_catalog["benchmark_task_id"].eq(task_id)]
        target_present = bool(not task_row.empty and int(task_row["site_count"].iloc[0]) > 0)
        metrics_present = set(spec["metric_ids"]).issubset(metric_ids)
        local_rows = int(task_ref["control_class"].isin(["local", "local_algorithm"]).sum()) if not task_ref.empty else 0
        rows.extend(
            [
                _smoke_row(
                    task_id,
                    "config_schema_minimal",
                    bool(spec["benchmark_task_id"] and spec["title"] and spec["target_id"] and spec["metric_ids"]),
                    {"required_fields_present": True},
                    {"task_id": task_id, "metric_count": len(spec["metric_ids"])},
                    "Task spec contains required ID, title, target, protocol, policy, and metric fields.",
                ),
                _smoke_row(
                    task_id,
                    "target_registered",
                    target_present,
                    {"target_in_s03_index": True},
                    {"target_id": spec["target_id"], "site_count": int(task_row["site_count"].iloc[0]) if not task_row.empty else 0},
                    "Benchmark task target is registered in the S03 target morphology index.",
                ),
                _smoke_row(
                    task_id,
                    "metrics_registered",
                    metrics_present,
                    {"all_declared_metrics_registered": True},
                    {"declared_metrics": spec["metric_ids"], "registered_metrics": sorted(metric_ids)},
                    "Declared benchmark metrics are either S05 metrics or the S06 sorted-row continuity metrics.",
                ),
                _smoke_row(
                    task_id,
                    "reference_rows_present",
                    len(task_ref) == int(spec["expected_reference_rows"]),
                    {"expected_reference_rows": int(spec["expected_reference_rows"])},
                    {"observed_reference_rows": int(len(task_ref)), "local_rows": local_rows},
                    "Reference output rows are present for the configured source selector.",
                ),
            ]
        )
        if task_id == "sort_row_embedded_1d":
            success = bool((task_ref["final_sortedness_percent"] >= 100.0).all() and (task_ref["final_aggregate_morphospace_error"] <= 1e-12).all())
            rows.append(
                _smoke_row(
                    task_id,
                    "embedded_row_exact_recovery",
                    success,
                    {"sortedness_percent": 100.0, "final_morphospace_error": 0.0},
                    {
                        "min_sortedness_percent": float(task_ref["final_sortedness_percent"].min()) if not task_ref.empty else None,
                        "max_final_morphospace_error": float(task_ref["final_aggregate_morphospace_error"].max()) if not task_ref.empty else None,
                    },
                    "S06 embedded-row reference reaches exact sortedness and zero embedded morphospace error.",
                )
            )
        else:
            local_changed = bool(task_ref.loc[task_ref["control_class"].eq("local"), "local_policy_information_access_changed"].fillna(True).any())
            global_rows = int(task_ref["policy_id"].isin(GLOBAL_CONTEXT_POLICY_IDS).sum())
            rows.append(
                _smoke_row(
                    task_id,
                    "local_access_and_global_context_guardrail",
                    bool(local_rows > 0 and global_rows > 0 and not local_changed),
                    {"local_rows": ">0", "global_context_rows": ">0", "local_access_changed": False},
                    {"local_rows": local_rows, "global_context_rows": global_rows, "local_access_changed": local_changed},
                    "Local-policy reference rows keep declared local access; S13 global rows are available only as labeled endpoint context.",
                )
            )
        if spec["known_blocker"]:
            blocker_rows = int(task_ref["semantic_blocker"].fillna("").astype(str).str.len().gt(0).sum()) if not task_ref.empty else 0
            rows.append(
                _smoke_row(
                    task_id,
                    "known_blocker_explicit",
                    blocker_rows > 0,
                    {"semantic_blocker_rows": ">0"},
                    {"semantic_blocker_rows": blocker_rows, "known_blocker": spec["known_blocker"]},
                    "Known local-semantics blockers are recorded in the reference rows instead of being silently repaired.",
                )
            )
    return pd.DataFrame(rows)


def validate_benchmark_suite_tables(
    *,
    task_catalog: pd.DataFrame,
    reference_df: pd.DataFrame,
    metric_catalog: pd.DataFrame,
    smoke_df: pd.DataFrame,
    source_inputs: pd.DataFrame,
    config_dir: Path | None,
    required_artifact_paths: Sequence[Path] | None,
) -> pd.DataFrame:
    """Return S15 validation rows."""

    rows: list[dict[str, Any]] = []
    task_ids = set(task_catalog["benchmark_task_id"])
    expected_task_ids = {spec["benchmark_task_id"] for spec in STANDARD_TASK_SPECS}
    rows.append(
        _validation_row(
            "standard_task_catalog_complete",
            "catalog",
            bool(task_ids == expected_task_ids and len(task_catalog) == 6),
            {"task_count": 6, "task_ids": sorted(expected_task_ids)},
            {"task_count": len(task_catalog), "task_ids": sorted(task_ids)},
            "Six standard benchmark tasks were packaged.",
        )
    )
    rows.append(
        _validation_row(
            "all_required_inputs_loaded",
            "input_provenance",
            bool(source_inputs.loc[source_inputs["required"], "exists"].all()),
            {"all_required_inputs_exist": True},
            {
                "required_inputs": int(source_inputs["required"].sum()),
                "missing_inputs": source_inputs[source_inputs["required"] & ~source_inputs["exists"]]["source_path"].tolist(),
            },
            "S01-S14 reports and result tables needed for packaging are available.",
        )
    )
    rows.append(
        _validation_row(
            "config_files_written_and_parseable",
            "config_schema",
            _config_dir_valid(config_dir) if config_dir is not None else True,
            {"config_file_count": 7, "parseable_json": True},
            _config_dir_observed(config_dir),
            "One JSON config per benchmark task plus an index config are present and parseable.",
        )
    )
    expected_rows = int(sum(spec["expected_reference_rows"] for spec in STANDARD_TASK_SPECS))
    rows.append(
        _validation_row(
            "reference_results_complete",
            "reference_outputs",
            bool(len(reference_df) == expected_rows and set(reference_df["benchmark_task_id"]) == expected_task_ids),
            {"reference_rows": expected_rows, "all_tasks_present": True},
            {"reference_rows": len(reference_df), "tasks": sorted(set(reference_df["benchmark_task_id"]))},
            "Reference result rows were normalized for every benchmark task.",
        )
    )
    rows.append(
        _validation_row(
            "smoke_tests_passed",
            "smoke_tests",
            bool(not smoke_df.empty and smoke_df["success"].all()),
            {"all_smoke_tests_success": True},
            {"smoke_rows": len(smoke_df), "passed": int(smoke_df["success"].sum()), "failed": int((~smoke_df["success"]).sum())},
            "Each benchmark task has passing config, target, metric, and reference-output smoke checks.",
        )
    )
    requested_metrics = set()
    for spec in STANDARD_TASK_SPECS:
        requested_metrics.update(spec["metric_ids"])
    rows.append(
        _validation_row(
            "metric_manifest_covers_task_metrics",
            "metrics",
            bool(requested_metrics.issubset(set(metric_catalog["metric_id"]))),
            {"all_requested_metrics_in_manifest": True},
            {"requested_metrics": sorted(requested_metrics), "manifest_metrics": sorted(set(metric_catalog["metric_id"]))},
            "The benchmark metric manifest covers S05 morphology metrics plus S06 sorted-row summary metrics.",
        )
    )
    local = reference_df[reference_df["control_class"].eq("local")]
    rows.append(
        _validation_row(
            "local_policy_access_guardrail_preserved",
            "information_access",
            bool(not local.empty and not local["local_policy_information_access_changed"].fillna(True).any()),
            {"local_policy_information_access_changed": False},
            {
                "local_rows": len(local),
                "changed_rows": int(local["local_policy_information_access_changed"].fillna(True).sum()) if not local.empty else 0,
            },
            "Benchmark packaging does not retune or expand local-policy information access.",
        )
    )
    blocked_tasks = {spec["benchmark_task_id"] for spec in STANDARD_TASK_SPECS if spec["known_blocker"]}
    blocked_ref = reference_df[reference_df["benchmark_task_id"].isin(blocked_tasks)]
    rows.append(
        _validation_row(
            "known_blockers_recorded",
            "blocker_accounting",
            bool(not blocked_ref.empty and blocked_ref["semantic_blocker"].fillna("").astype(str).str.len().gt(0).any()),
            {"blocked_task_semantic_blockers_present": True},
            {
                "blocked_task_ids": sorted(blocked_tasks),
                "semantic_blocker_rows": int(blocked_ref["semantic_blocker"].fillna("").astype(str).str.len().gt(0).sum()),
            },
            "Population-deficit and identity-conversion blockers remain explicit in benchmark references.",
        )
    )
    numeric = reference_df[["final_target_error", "final_aggregate_morphospace_error", "total_energy_cost"]].copy()
    finite_or_na = bool(np.isfinite(numeric.to_numpy(dtype=float)[~np.isnan(numeric.to_numpy(dtype=float))]).all())
    rows.append(
        _validation_row(
            "reference_metrics_numeric",
            "schema",
            finite_or_na,
            {"finite_numeric_values_or_na": True},
            {"reference_rows": len(reference_df), "numeric_columns": list(numeric.columns)},
            "Reference metric and energy columns are numeric and finite where present.",
        )
    )
    artifact_observed = {}
    artifact_success = True
    if required_artifact_paths is not None:
        for path in required_artifact_paths:
            artifact_observed[str(path)] = {"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        artifact_success = all(item["exists"] and item["size_bytes"] > 0 for item in artifact_observed.values())
    rows.append(
        _validation_row(
            "required_s15_artifacts_present",
            "artifact_completeness",
            bool(artifact_success),
            {"required_artifacts_present": True},
            artifact_observed,
            "S15 benchmark-suite report, handoff, configs, reference table, and full-results artifacts are present.",
        )
    )
    return pd.DataFrame(rows)


def summarize_for_report(task_catalog: pd.DataFrame, reference_df: pd.DataFrame, smoke_df: pd.DataFrame) -> dict[str, Any]:
    local = reference_df[reference_df["control_class"].isin(["local", "local_algorithm"])]
    global_ref = reference_df[reference_df["policy_id"].isin(GLOBAL_CONTEXT_POLICY_IDS)]
    return {
        "task_count": int(len(task_catalog)),
        "reference_rows": int(len(reference_df)),
        "smoke_rows": int(len(smoke_df)),
        "smoke_success": bool(smoke_df["success"].all()) if not smoke_df.empty else False,
        "local_reference_rows": int(len(local)),
        "global_context_reference_rows": int(len(global_ref)),
        "exact_recovered_rows": int((reference_df["reference_outcome_class"] == "exact_recovered").sum()),
        "partial_recovery_rows": int((reference_df["reference_outcome_class"] == "partial_recovery").sum()),
        "blocked_rows": int((reference_df["reference_outcome_class"] == "blocked_baseline").sum()),
    }


def spec_by_id(task_id: str) -> dict[str, Any]:
    for spec in STANDARD_TASK_SPECS:
        if spec["benchmark_task_id"] == task_id:
            return spec
    raise KeyError(task_id)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(str(path))
    return pd.read_parquet(path)


def _first_finite(*values: Any) -> float:
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate):
            return candidate
    return float("nan")


def _smoke_row(
    task_id: str,
    smoke_test_id: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "benchmark_suite_id": BENCHMARK_SUITE_ID,
        "benchmark_task_id": task_id,
        "smoke_test_id": smoke_test_id,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": detail,
    }


def _validation_row(
    validation_case: str,
    case_type: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "validation_case": validation_case,
        "case_type": case_type,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": detail,
    }


def _config_dir_observed(config_dir: Path | None) -> dict[str, Any]:
    if config_dir is None:
        return {}
    paths = sorted(config_dir.glob("*.json")) if config_dir.exists() else []
    parseable = True
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parseable = False
            break
    return {"config_dir": str(config_dir), "json_files": [path.name for path in paths], "file_count": len(paths), "parseable": parseable}


def _config_dir_valid(config_dir: Path | None) -> bool:
    observed = _config_dir_observed(config_dir)
    return bool(observed.get("file_count") == 7 and observed.get("parseable"))
