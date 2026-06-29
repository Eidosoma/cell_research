from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .goal_catalog import GOAL_CATALOG_VERSION, GOAL_SCHEMA_VERSION
from .policy_catalog import POLICY_CATALOG_VERSION, POLICY_SCHEMA_VERSION, read_table, safe_json_loads
from .world_schema import CATALOG_VERSION as WORLD_CATALOG_VERSION
from .world_schema import SCHEMA_VERSION as WORLD_SCHEMA_VERSION
from .world_schema import sha256_path


BEHAVIOR_SCHEMA_VERSION = "e07_s04_behavior_corpus.v1"
BEHAVIOR_CORPUS_VERSION = "e07_s04_unified_behavior_corpus.v1"
BEHAVIOR_CLAIM_BOUNDARY = (
    "Computational behavior corpus only; records link simulated policies, worlds, goals, perturbations, "
    "and metrics from upstream toy-model experiments. These rows are not direct biological, clinical, "
    "cognitive, sentience, or living-chimera evidence."
)

REQUIRED_BEHAVIOR_FIELDS = (
    "behaviorRecordId",
    "behaviorSchemaVersion",
    "behaviorCorpusVersion",
    "sourceExperimentId",
    "sourceStepId",
    "sourceTableName",
    "sourceTablePath",
    "sourceTableSha256",
    "sourceRowIndex",
    "sourceRowHash",
    "worldId",
    "worldResolutionStatus",
    "candidateWorldIdsJson",
    "abstractGoalId",
    "goalResolutionStatus",
    "candidateGoalIdsJson",
    "abstractPolicyIdsJson",
    "policyResolutionStatus",
    "candidatePolicyIdsJson",
    "runId",
    "seedJson",
    "schedulerJson",
    "perturbationJson",
    "metricValuesJson",
    "behaviorVectorJson",
    "finalStateJson",
    "trajectorySummaryJson",
    "rawTraceLinksJson",
    "missingnessJson",
    "claimBoundary",
)

JSON_COLUMNS = (
    "candidateWorldIdsJson",
    "candidateGoalIdsJson",
    "abstractPolicyIdsJson",
    "candidatePolicyIdsJson",
    "seedJson",
    "schedulerJson",
    "perturbationJson",
    "metricValuesJson",
    "behaviorVectorJson",
    "finalStateJson",
    "trajectorySummaryJson",
    "rawTraceLinksJson",
    "missingnessJson",
)

VALIDATION_REQUIRED_EXPERIMENTS = {f"E{i:02d}" for i in range(1, 7)}

SOURCE_TABLE_SPECS: tuple[dict[str, str], ...] = (
    {"experimentId": "E01", "relativePath": "results/e01_results.parquet", "sourceKind": "baseline_summary"},
    {"experimentId": "E01", "relativePath": "results/e01_s04_replicate_summary.parquet", "sourceKind": "run_summary"},
    {"experimentId": "E01", "relativePath": "results/e01_frozen_cell_robustness.parquet", "sourceKind": "run_summary"},
    {"experimentId": "E01", "relativePath": "results/e01_delayed_gratification_summary.parquet", "sourceKind": "metric_summary"},
    {"experimentId": "E01", "relativePath": "results/e01_same_goal_chimeras.parquet", "sourceKind": "chimera_runs"},
    {"experimentId": "E01", "relativePath": "results/e01_duplicate_value_chimeras.parquet", "sourceKind": "chimera_runs"},
    {"experimentId": "E01", "relativePath": "results/e01_opposite_direction_chimeras.parquet", "sourceKind": "chimera_runs"},
    {"experimentId": "E01", "relativePath": "results/e01_scaled_replication_summary.parquet", "sourceKind": "scaled_summary"},
    {"experimentId": "E01", "relativePath": "results/e01_statistics.parquet", "sourceKind": "statistics"},
    {"experimentId": "E01", "relativePath": "results/e01_aggregation_peak_summary.parquet", "sourceKind": "aggregation_summary"},
    {"experimentId": "E02", "relativePath": "results/e02_scheduler_regimes.parquet", "sourceKind": "scheduler_control"},
    {"experimentId": "E02", "relativePath": "results/e02_activation_rates.parquet", "sourceKind": "activation_control"},
    {"experimentId": "E02", "relativePath": "results/e02_label_shuffle_observed.parquet", "sourceKind": "label_shuffle_control"},
    {"experimentId": "E02", "relativePath": "results/e02_local_move_null_summary.parquet", "sourceKind": "null_summary"},
    {"experimentId": "E02", "relativePath": "results/e02_dg_observed.parquet", "sourceKind": "dg_observed"},
    {"experimentId": "E02", "relativePath": "results/e02_dg_null_summary.parquet", "sourceKind": "dg_null_summary"},
    {"experimentId": "E02", "relativePath": "results/e02_alternative_metrics.parquet", "sourceKind": "metric_audit"},
    {"experimentId": "E02", "relativePath": "results/e02_frozen_placement.parquet", "sourceKind": "frozen_control"},
    {"experimentId": "E02", "relativePath": "results/e02_frozen_behavior_variants.parquet", "sourceKind": "frozen_control"},
    {"experimentId": "E02", "relativePath": "results/e02_stop_condition_sensitivity.parquet", "sourceKind": "stop_control"},
    {"experimentId": "E02", "relativePath": "results/e02_strong_statistics.parquet", "sourceKind": "statistics"},
    {"experimentId": "E02", "relativePath": "results/e02_claim_survival_table.parquet", "sourceKind": "claim_audit"},
    {"experimentId": "E02", "relativePath": "results/e02_speed_matched_algotypes.parquet", "sourceKind": "speed_control"},
    {"experimentId": "E02", "relativePath": "results/e02_input_distributions.parquet", "sourceKind": "input_control"},
    {"experimentId": "E02", "relativePath": "results/e02_dummy_algotypes.parquet", "sourceKind": "dummy_control"},
    {"experimentId": "E03", "relativePath": "results/e03_coarse_sweep.parquet", "sourceKind": "policy_sweep"},
    {"experimentId": "E03", "relativePath": "results/e03_morphospace_metrics.parquet", "sourceKind": "policy_sweep"},
    {"experimentId": "E03", "relativePath": "results/e03_feature_ablations.parquet", "sourceKind": "ablation_summary"},
    {"experimentId": "E03", "relativePath": "results/e03_phase_boundaries.parquet", "sourceKind": "phase_boundary_summary"},
    {"experimentId": "E03", "relativePath": "results/e03_s07_competence_summary.parquet", "sourceKind": "competence_summary"},
    {"experimentId": "E03", "relativePath": "results/e03_s09_phase_boundary_runs.parquet", "sourceKind": "phase_boundary_runs"},
    {"experimentId": "E03", "relativePath": "results/e03_s12_ablation_runs.parquet", "sourceKind": "ablation_runs"},
    {"experimentId": "E03", "relativePath": "results/e03_s13_same_seed_runs.parquet", "sourceKind": "classic_comparison_runs"},
    {"experimentId": "E03", "relativePath": "results/e03_s14_frontier_validation_runs.parquet", "sourceKind": "frontier_validation_runs"},
    {"experimentId": "E03", "relativePath": "results/e03_classics_vs_discovered.parquet", "sourceKind": "behavior_atlas_summary"},
    {"experimentId": "E04", "relativePath": "results/e04_memory_ablations.parquet", "sourceKind": "memory_ablation_runs"},
    {"experimentId": "E04", "relativePath": "results/e04_communication_ablations.parquet", "sourceKind": "communication_ablation_runs"},
    {"experimentId": "E04", "relativePath": "results/e04_s11_competence_proxy_results.parquet", "sourceKind": "competence_proxy_runs"},
    {"experimentId": "E04", "relativePath": "results/e04_centralized_comparison.parquet", "sourceKind": "centralized_comparison"},
    {"experimentId": "E04", "relativePath": "results/e04_overfitting_transfer.parquet", "sourceKind": "transfer_runs"},
    {"experimentId": "E04", "relativePath": "results/e04_evolution_runs.parquet", "sourceKind": "evolution_summary"},
    {"experimentId": "E04", "relativePath": "results/e04_s03_repair_task_run_summary.parquet", "sourceKind": "repair_runs"},
    {"experimentId": "E04", "relativePath": "results/e04_s05_trivial_controller_results.parquet", "sourceKind": "homeostasis_baseline"},
    {"experimentId": "E04", "relativePath": "results/e04_s08_elite_cpu_replays.parquet", "sourceKind": "elite_replays"},
    {"experimentId": "E04", "relativePath": "results/e04_s15_selected_policy_handoff_replays.parquet", "sourceKind": "handoff_replays"},
    {"experimentId": "E05", "relativePath": "results/e05_1d_embedded_replication.parquet", "sourceKind": "embedded_1d_runs"},
    {"experimentId": "E05", "relativePath": "results/e05_gpu_sweeps.parquet", "sourceKind": "gpu_morphology_runs"},
    {"experimentId": "E05", "relativePath": "results/e05_higher_dimensional_dg.parquet", "sourceKind": "higher_dimensional_dg"},
    {"experimentId": "E05", "relativePath": "results/e05_local_global_control.parquet", "sourceKind": "local_global_control"},
    {"experimentId": "E05", "relativePath": "results/e05_morphology_benchmarks.parquet", "sourceKind": "benchmark_runs"},
    {"experimentId": "E05", "relativePath": "results/e05_morphospace_trajectories.parquet", "sourceKind": "trajectory_summary"},
    {"experimentId": "E05", "relativePath": "results/e05_regeneration_tests.parquet", "sourceKind": "regeneration_runs"},
    {"experimentId": "E05", "relativePath": "results/e05_scaling_tests.parquet", "sourceKind": "scaling_runs"},
    {"experimentId": "E05", "relativePath": "results/e05_scrambled_embryo_tests.parquet", "sourceKind": "scrambled_recovery_runs"},
    {"experimentId": "E05", "relativePath": "results/e05_symmetry_breaking.parquet", "sourceKind": "symmetry_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_compatibility_metrics.parquet", "sourceKind": "compatibility_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_dominance_contests.parquet", "sourceKind": "dominance_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_dominance_mosaics.parquet", "sourceKind": "mosaic_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_goal_compatibility.parquet", "sourceKind": "goal_compatibility_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_governance_mechanisms.parquet", "sourceKind": "governance_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_governance_interventions.parquet", "sourceKind": "governance_intervention_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_intervention_search.parquet", "sourceKind": "intervention_search"},
    {"experimentId": "E06", "relativePath": "results/e06_mixture_ratios.parquet", "sourceKind": "mixture_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_mixture_sweeps.parquet", "sourceKind": "mixture_sweeps"},
    {"experimentId": "E06", "relativePath": "results/e06_initial_arrangements.parquet", "sourceKind": "arrangement_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_interface_rules.parquet", "sourceKind": "interface_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_graft_experiments.parquet", "sourceKind": "graft_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_mutant_clone_experiments.parquet", "sourceKind": "mutant_clone_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_developmental_history.parquet", "sourceKind": "history_runs"},
    {"experimentId": "E06", "relativePath": "results/e06_phase_diagram_summary.parquet", "sourceKind": "phase_summary"},
    {"experimentId": "E06", "relativePath": "results/e06_s01_pure_policy_validation.parquet", "sourceKind": "pure_policy_validation"},
)

CONTEXT_EXCLUDE_TOKENS = (
    "id",
    "hash",
    "json",
    "path",
    "version",
    "label",
    "title",
    "description",
    "family",
    "kind",
    "class",
    "category",
    "source",
    "artifact",
    "schema",
    "formula",
    "comparison",
    "note",
    "rationale",
    "claim",
)

METRIC_INCLUDE_TOKENS = (
    "sorted",
    "monotonic",
    "complete",
    "success",
    "error",
    "repair",
    "recover",
    "remaining",
    "energy",
    "target",
    "earth",
    "boundary",
    "topology",
    "symmetry",
    "aggregation",
    "dg",
    "delayed",
    "dominance",
    "compatibility",
    "governance",
    "intervention",
    "effect",
    "pvalue",
    "qvalue",
    "ci",
    "mean",
    "sd",
    "sem",
    "rate",
    "ratio",
    "score",
    "fitness",
    "reward",
    "penalty",
    "swap",
    "comparison",
    "activation",
    "event",
    "blocked",
    "failure",
    "duration",
    "distance",
    "auc",
    "curvature",
    "entropy",
    "gini",
    "final",
    "initial",
    "delta",
    "relative",
    "transfer",
    "overfit",
    "calibration",
    "verdict",
    "outcome",
)

BEHAVIOR_VECTOR_SOURCES: dict[str, tuple[str, ...]] = {
    "initial_sortedness_percent": (
        "initial_sortedness_percent",
        "initialSortednessPercent",
        "initialGoalSortednessPercent",
        "initialSortednessRaw",
    ),
    "final_sortedness_percent": (
        "final_sortedness_percent",
        "finalSortednessPercent",
        "finalGoalSortednessPercent",
        "finalIncreasingSortednessPercent",
        "finalSortednessScore",
        "realFinalSortednessMean",
    ),
    "final_monotonicity_error": ("final_monotonicity_error", "finalMonotonicityError", "monotonicityError"),
    "completed": ("completed", "completionSuccess", "runSucceeded", "completedGoal", "benchmark_success"),
    "swap_count": ("swap_count", "swapCount", "mean_steps", "swapCountMean"),
    "comparison_count": ("comparison_count", "comparisonCount", "activationComparisonCount"),
    "activation_count": ("activation_count", "activationCount", "realizedActivationCount"),
    "event_count": ("event_count", "eventCount", "trajectoryEventCount"),
    "blocked_move_attempts": ("blockedMoveAttempts", "frozenSwapAttempts", "blockedSwapAttempts"),
    "delayed_gratification": (
        "meanDelayedGratification",
        "delayedGratification",
        "observedDgMean",
        "observedMinusNullDgMean",
        "dgMaxDrop",
        "realDgMaxDropMean",
    ),
    "aggregation": (
        "finalAggregationMean",
        "peakAggregationMean",
        "meanCurvePeakAggregation",
        "realPeakAggregationMean",
        "aggregationFinal",
        "finalAggregation",
    ),
    "target_error": (
        "final_composite_error",
        "finalTargetEnergy",
        "final_target_energy",
        "final_error_proxy",
        "target_energy",
        "final_mean_error",
    ),
    "error_reduction": ("relative_error_reduction", "delta_composite_error", "relativeErrorReduction"),
    "repair_success": ("repair_success", "regeneration_success", "recoveredCellCount", "remainingFrozenCellCount"),
    "symmetry_success": ("final_symmetry_success", "symmetry_breaking_success", "initial_symmetry_success"),
    "compatibility": ("meanPolicyGoalScore", "compatibilityScore", "meanCompatibility", "pairCompatibilityScore"),
    "dominance": ("dominanceScore", "dominanceProxy", "winnerMargin", "dominanceGini"),
    "homeostasis": ("timeInRangeFraction", "failureDurationTicks", "unrecoveredPerturbationCount"),
    "fitness_or_score": ("proxyFitness", "score", "benchmark_score", "competenceScore"),
}


def json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, (list, tuple, dict, set, str, Path)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def json_hash(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text.upper()


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _not_missing(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return True


def _float_or_none(value: Any) -> float | None:
    if not _not_missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_parse(value: Any, default: Any = None) -> Any:
    parsed = safe_json_loads(value, default)
    return default if parsed is None else parsed


def _as_list(value: Any) -> list[Any]:
    if not _not_missing(value):
        return []
    parsed = _safe_parse(value, None)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return list(parsed)
    if isinstance(value, str) and "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _unique(values: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _value_text(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    return " ".join(str(row.get(column, "")) for column in columns if _not_missing(row.get(column))).lower()


def _world_ids(worlds: pd.DataFrame, experiment_id: str) -> list[str]:
    return sorted(worlds.loc[worlds["experimentId"].eq(experiment_id), "worldId"].astype(str).unique())


def _world_by_step(worlds: pd.DataFrame, experiment_id: str, step_tokens: Sequence[str]) -> list[str]:
    wanted = {str(token).replace("E" + experiment_id[-2:] + "_", "").upper() for token in step_tokens}
    hits: list[str] = []
    for _, row in worlds[worlds["experimentId"].eq(experiment_id)].iterrows():
        steps = {str(item).upper() for item in _safe_parse(row.get("sourceStepIds"), []) or []}
        if steps & wanted:
            hits.append(str(row["worldId"]))
    return sorted(set(hits))


def _first_existing_id(candidate_ids: Sequence[str], valid_ids: set[str]) -> str | None:
    for candidate in candidate_ids:
        if candidate in valid_ids:
            return candidate
    return None


def _infer_e01_world(row: Mapping[str, Any], table_name: str, valid_worlds: set[str]) -> str:
    text = _value_text(row, row.keys()) + " " + table_name
    if "opposite" in table_name or "opposite" in text:
        return "E01_S12_opposite_direction_chimera_sorting"
    if "duplicate" in table_name or "duplicate" in text:
        return "E01_S11_duplicate_value_chimera_sorting"
    if "same_goal" in table_name or "same goal" in text:
        return "E01_S09_same_goal_chimera_sorting"
    if "frozen" in table_name or _float_or_none(row.get("frozenCount")) not in (None, 0.0):
        return "E01_S07_frozen_cell_sorting"
    return _first_existing_id(["E01_S04_1d_sorting_baseline"], valid_worlds) or sorted(valid_worlds)[0]


def _infer_e02_world(row: Mapping[str, Any], table_name: str, valid_worlds: set[str]) -> str:
    if "scheduler" in table_name:
        return "E02_S02_scheduler_regime_worlds"
    if "activation" in table_name:
        return "E02_S03_activation_rate_worlds"
    if any(token in table_name for token in ("label_shuffle", "local_move", "speed_matched", "dummy")):
        return "E02_S04_S07_aggregation_null_worlds"
    if "dg" in table_name:
        return "E02_S08_delayed_gratification_null_worlds"
    if any(token in table_name for token in ("alternative_metrics", "input_distributions")):
        return "E02_S09_S10_metric_distribution_worlds"
    if any(token in table_name for token in ("frozen", "stop_condition", "strong_statistics", "claim_survival")):
        return "E02_S11_S13_frozen_and_stop_variant_worlds"
    return _first_existing_id(["E02_S01_deterministic_reference_sorting"], valid_worlds) or sorted(valid_worlds)[0]


def _infer_e03_world(row: Mapping[str, Any], table_name: str, valid_worlds: set[str]) -> str:
    if any(token in table_name for token in ("coarse_sweep", "s07", "s09", "phase_boundary_runs")):
        return "E03_S06_S09_batch_sweep_and_quality_diversity_world"
    if any(token in table_name for token in ("feature_ablations", "ablation_runs", "same_seed", "frontier", "classics_vs_discovered", "morphospace_metrics")):
        return "E03_S10_S15_behavior_atlas_world"
    return _first_existing_id(["E03_S02_S05_policy_dsl_corpus_world"], valid_worlds) or sorted(valid_worlds)[0]


def _infer_e04_world(row: Mapping[str, Any], table_name: str, valid_worlds: set[str]) -> str:
    if "memory" in table_name:
        return "E04_S01_memory_augmented_sorting_world"
    if "communication" in table_name:
        return "E04_S02_signal_communication_world"
    if "repair_task" in table_name:
        return "E04_S03_repairable_frozen_world"
    if "trivial_controller" in table_name:
        return "E04_S04_S05_homeostasis_and_damage_world"
    if any(token in table_name for token in ("competence", "centralized", "overfitting", "evolution", "elite", "handoff")):
        return "E04_S06_S15_learning_evolution_synthesis_world"
    return _first_existing_id(["E04_S06_S15_learning_evolution_synthesis_world"], valid_worlds) or sorted(valid_worlds)[0]


def _infer_e05_world(row: Mapping[str, Any], table_name: str, valid_worlds: set[str]) -> str:
    benchmark_id = row.get("benchmark_id") or row.get("benchmarkId")
    if _not_missing(benchmark_id):
        hit = f"E05_S15_benchmark_{_norm(benchmark_id)}"
        if hit in valid_worlds:
            return hit
    task_id = row.get("task_id") or row.get("taskId")
    if _not_missing(task_id):
        hit = f"E05_S10_symmetry_{_norm(task_id)}"
        if hit in valid_worlds:
            return hit
    perturbation_id = row.get("perturbation_id") or row.get("perturbationId")
    if _not_missing(perturbation_id) and any(token in table_name for token in ("regeneration", "scrambled")):
        hit = f"E05_S08_perturbation_{_norm(perturbation_id)}"
        if hit in valid_worlds:
            return hit
    target_id = row.get("target_id") or row.get("targetId")
    if _not_missing(target_id):
        hit = f"E05_S03_target_{_norm(target_id)}"
        if hit in valid_worlds:
            return hit
    if "1d_embedded" in table_name:
        return "E05_S03_target_sorted_row_8"
    if "symmetry" in table_name:
        return _first_existing_id(
            [
                "E05_S10_symmetry_axis_gradient_9x9",
                "E05_S10_symmetry_ring_from_neutral_9x9",
                "E05_S10_symmetry_asymmetric_appendage_9x9",
            ],
            valid_worlds,
        ) or sorted(valid_worlds)[0]
    if "benchmark" in table_name:
        return _first_existing_id(["E05_S15_benchmark_sort_row"], valid_worlds) or sorted(valid_worlds)[0]
    if any(token in table_name for token in ("regeneration", "scrambled")):
        return _first_existing_id(["E05_S08_perturbation_contiguous_chunk_removal"], valid_worlds) or sorted(valid_worlds)[0]
    return _first_existing_id(["E05_S03_target_sorted_row_8"], valid_worlds) or sorted(valid_worlds)[0]


def _infer_e06_world(row: Mapping[str, Any], table_name: str, valid_worlds: set[str]) -> str:
    if "pure_policy" in table_name:
        return "E06_S01_pure_policy_panel_world"
    if any(token in table_name for token in ("mixture", "initial_arrangements")):
        return "E06_S02_S03_mixture_arrangement_worlds"
    if any(token in table_name for token in ("compatibility_metrics", "goal_compatibility")):
        return "E06_S04_S05_goal_compatibility_worlds"
    if any(token in table_name for token in ("dominance", "mosaic", "interface")):
        return "E06_S06_S08_dominance_mosaic_interface_worlds"
    if any(token in table_name for token in ("governance", "intervention", "graft", "mutant", "developmental_history")):
        return "E06_S09_S14_governance_graft_mutant_history_intervention_worlds"
    if "phase_diagram" in table_name:
        return "E06_S15_chimeric_playbook_world_panel"
    return _first_existing_id(["E06_S02_S03_mixture_arrangement_worlds"], valid_worlds) or sorted(valid_worlds)[0]


def infer_world_id(
    row: Mapping[str, Any],
    experiment_id: str,
    table_name: str,
    worlds: pd.DataFrame,
    world_ids_by_experiment: Mapping[str, set[str]] | None = None,
) -> tuple[str, str, list[str]]:
    valid_worlds = set(world_ids_by_experiment.get(experiment_id, set())) if world_ids_by_experiment else set(_world_ids(worlds, experiment_id))
    if not valid_worlds:
        return "", "unresolved_no_experiment_worlds", []
    mapper = {
        "E01": _infer_e01_world,
        "E02": _infer_e02_world,
        "E03": _infer_e03_world,
        "E04": _infer_e04_world,
        "E05": _infer_e05_world,
        "E06": _infer_e06_world,
    }.get(experiment_id)
    world_id = mapper(row, table_name, valid_worlds) if mapper else sorted(valid_worlds)[0]
    status = "exact_or_family_rule"
    if world_id not in valid_worlds:
        world_id = sorted(valid_worlds)[0]
        status = "fallback_experiment_family"
    candidates = [world_id, *[item for item in sorted(valid_worlds) if item != world_id]]
    return world_id, status, candidates


class PolicyResolver:
    def __init__(self, policies: pd.DataFrame) -> None:
        self.policies = policies
        self.valid_ids = set(policies["abstractPolicyId"].astype(str))
        self.by_key: dict[str, list[str]] = defaultdict(list)
        self.by_experiment: dict[str, list[str]] = defaultdict(list)
        self.by_world: dict[str, list[str]] = defaultdict(list)
        for _, row in policies.iterrows():
            abstract_id = str(row["abstractPolicyId"])
            source_exp = str(row.get("sourceExperimentId", ""))
            if source_exp:
                self.by_experiment[source_exp].append(abstract_id)
            for field in ("abstractPolicyId", "sourcePolicyId", "policyLabel"):
                value = row.get(field)
                if _not_missing(value):
                    for key in self._keys(value):
                        self.by_key[key].append(abstract_id)
            for world_id in _safe_parse(row.get("linkedWorldIds"), []) or []:
                self.by_world[str(world_id)].append(abstract_id)

    def _keys(self, value: Any) -> list[str]:
        text = str(value)
        keys = {_norm(text), text.strip().lower()}
        if text.startswith(("E01::", "E02::", "E03::", "E04::", "E05::", "E06::")):
            keys.add(_norm(text.split("::")[-1]))
        return [key for key in keys if key]

    def resolve(
        self,
        row: Mapping[str, Any],
        experiment_id: str,
        world_id: str,
        candidate_goal_policy_ids: Sequence[str] | None = None,
        policy_columns: Sequence[str] | None = None,
    ) -> tuple[list[str], str, list[str], list[str]]:
        raw_tokens = extract_policy_tokens(row, experiment_id, policy_columns)
        exact: list[str] = []
        for token in raw_tokens:
            for key in self._keys(token):
                exact.extend(self.by_key.get(key, []))
        exact.extend(self._classic_candidates(row, experiment_id))
        exact = _unique(item for item in exact if item in self.valid_ids)
        if exact:
            return exact, "exact_policy_id_or_source_token", _unique(raw_tokens, limit=50), exact[:25]

        candidates: list[str] = []
        seen: set[str] = set()
        for source in (candidate_goal_policy_ids or [], self.by_world.get(world_id, []), self.by_experiment.get(experiment_id, [])):
            for item in source:
                if item not in self.valid_ids or item in seen:
                    continue
                seen.add(item)
                candidates.append(item)
                if len(candidates) >= 30:
                    break
            if len(candidates) >= 30:
                break
        if candidates:
            return [], "candidate_from_world_or_goal", _unique(raw_tokens, limit=50), candidates
        return [], "unresolved_no_policy_token", _unique(raw_tokens, limit=50), []

    def _classic_candidates(self, row: Mapping[str, Any], experiment_id: str) -> list[str]:
        implementation = _norm(row.get("implementation"))
        algorithms: list[str] = []
        for key in ("algorithm", "algorithms", "basePolicy", "classicFamily"):
            for value in _as_list(row.get(key)):
                algorithms.append(_norm(value))
        hits: list[str] = []
        for algorithm in algorithms:
            if algorithm not in {"bubble", "insertion", "selection", "classic_bubble", "classic_insertion", "classic_selection"}:
                continue
            short = algorithm.replace("classic_", "")
            if experiment_id == "E01" or implementation in {"cell_view", ""}:
                hits.append(f"E01::cell_view_{short}")
            if implementation == "traditional":
                hits.append(f"E01::traditional_{short}")
            hits.append(f"E03::classic::classic_{short}")
            hits.append(f"E05::morphology::{short}")
            hits.append(f"E06::panel::e06_original_classic_{short}")
        return hits


def extract_policy_tokens(row: Mapping[str, Any], experiment_id: str, policy_columns: Sequence[str] | None = None) -> list[str]:
    tokens: list[str] = []
    direct_columns = (
        "abstractPolicyId",
        "sourcePolicyId",
        "policyId",
        "policy_id",
        "policy",
        "policyName",
        "policy_name",
        "policyIdsJson",
        "algorithm",
        "algorithms",
        "basePolicy",
        "controllerId",
        "controllerKind",
        "candidateId",
        "sourceCandidateId",
        "genomeId",
        "ablationPolicyId",
        "sourcePolicyId",
        "sourcePolicyIds",
        "originalPolicyId",
        "classicPolicyId",
        "discoveredPolicyId",
        "panelPolicyId",
        "leftPanelPolicyId",
        "rightPanelPolicyId",
        "thirdPanelPolicyId",
        "policyAPanelPolicyId",
        "policyBPanelPolicyId",
        "governanceRuleId",
        "interventionPolicyId",
    )
    for column in direct_columns:
        if column in row:
            tokens.extend(str(item) for item in _as_list(row.get(column)) if _not_missing(item))

    scan_columns = policy_columns if policy_columns is not None else list(row)
    for column in scan_columns:
        value = row.get(column)
        lower = str(column).lower()
        if "policy" not in lower and "controller" not in lower and "governance" not in lower:
            continue
        if "json" in lower:
            parsed = _safe_parse(value, None)
            if isinstance(parsed, list):
                tokens.extend(str(item) for item in parsed if _not_missing(item))
            elif isinstance(parsed, dict):
                tokens.extend(str(key) for key in parsed if _not_missing(key))
        elif lower.endswith("id") or lower.endswith("ids"):
            tokens.extend(str(item) for item in _as_list(value) if _not_missing(item))

    base_policy_spec = _safe_parse(row.get("basePolicySpecJson"), {})
    if isinstance(base_policy_spec, dict):
        for key in ("policy_id", "algotype", "family"):
            if _not_missing(base_policy_spec.get(key)):
                tokens.append(str(base_policy_spec[key]))

    return _unique(tokens, limit=100)


def _parse_goal_world_links(goals: pd.DataFrame) -> dict[str, list[str]]:
    by_world: dict[str, list[str]] = defaultdict(list)
    for _, row in goals.iterrows():
        goal_id = str(row["abstractGoalId"])
        for world_id in _safe_parse(row.get("linkedWorldIds"), []) or []:
            by_world[str(world_id)].append(goal_id)
    return {world_id: sorted(set(goal_ids)) for world_id, goal_ids in by_world.items()}


def _goal_policy_lookup(goals: pd.DataFrame) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for _, row in goals.iterrows():
        lookup[str(row["abstractGoalId"])] = [str(item) for item in (_safe_parse(row.get("linkedPolicyIds"), []) or [])]
    return lookup


def _goal_policy_candidates(goal_policy_lookup: Mapping[str, Sequence[str]], goal_ids: Sequence[str]) -> list[str]:
    if not goal_policy_lookup or not goal_ids:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for goal_id in goal_ids:
        for item in goal_policy_lookup.get(str(goal_id), []):
            text = str(item)
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= 100:
                return out
    return out


def infer_goal_id(
    row: Mapping[str, Any],
    experiment_id: str,
    table_name: str,
    world_id: str,
    goals: pd.DataFrame,
    world_to_goals: Mapping[str, Sequence[str]],
    valid_goal_ids: set[str] | None = None,
) -> tuple[str, str, list[str]]:
    valid_goals = valid_goal_ids if valid_goal_ids is not None else set(goals["abstractGoalId"].astype(str))
    candidates = _unique([str(item) for item in world_to_goals.get(world_id, [])])
    table_text = table_name.lower()
    row_text = _value_text(row, row.keys())

    explicit: list[str] = []
    benchmark_id = row.get("benchmark_id") or row.get("benchmarkId")
    if _not_missing(benchmark_id):
        explicit.append(f"G_BENCHMARK_{_slug(benchmark_id)}")
    target_id = row.get("target_id") or row.get("targetId")
    if _not_missing(target_id):
        explicit.append(f"G_TARGET_MORPHOLOGY_{_slug(target_id)}")
    task_id = row.get("task_id") or row.get("taskId")
    if _not_missing(task_id):
        explicit.append(f"G_SYMMETRY_{_slug(task_id)}")

    if experiment_id == "E06":
        if "opposite" in row_text:
            explicit.append("G_CHIMERA_GOAL_MODE_OPPOSITE_GOAL")
        if "same_goal" in row_text or "same goal" in row_text:
            explicit.append("G_CHIMERA_GOAL_MODE_SAME_GOAL")
        if any(token in table_text for token in ("dominance", "mosaic", "interface")):
            explicit.append("G_DOMINANCE_MOSAIC_INTERFACE_STABILITY")
        if any(token in table_text for token in ("governance", "intervention", "graft", "mutant", "history")):
            explicit.append("G_RESCUE_OR_CONTROL_INTERVENTION")
        if "phase_diagram" in table_text:
            explicit.append("G_CHIMERIC_CONTROL_PLAYBOOK_TRACEABILITY")
    if experiment_id == "E02" and ("null" in table_text or "shuffle" in table_text or "dummy" in table_text):
        explicit.append("G_NULL_OR_CONTROL_DISTRIBUTION")
    if experiment_id == "E04":
        if "repair" in table_text:
            explicit.append("G_RESTORE_ORDER_AFTER_DEFECT")
        if "trivial_controller" in table_text or "homeostasis" in table_text:
            explicit.append("G_HOMEOSTATIC_ORDER_MAINTENANCE")
        if any(token in table_text for token in ("competence", "centralized", "overfitting", "elite", "handoff")):
            explicit.append("G_REPAIR_COMPETENCE_PROXY")
    if experiment_id == "E05":
        if any(token in table_text for token in ("regeneration", "scaling", "scrambled")):
            explicit.append("G_REPAIR_TO_TARGET_MORPHOLOGY")
        if "higher_dimensional_dg" in table_text:
            explicit.append("G_BENCHMARK_HIGHER_DIMENSIONAL_DG_DIAGNOSTICS")
    if experiment_id == "E01":
        if "opposite" in table_text:
            explicit.append("G_CONFLICTING_DIRECTION_EQUILIBRIUM")
        if "duplicate" in table_text:
            explicit.append("G_NONDECREASING_DUPLICATE_ORDER")
        if "same_goal" in table_text:
            explicit.append("G_SHARED_GLOBAL_ORDER")

    for goal_id in explicit:
        if goal_id in valid_goals and (not candidates or goal_id in candidates or experiment_id == "E05"):
            return goal_id, "explicit_source_field_or_table_rule", _unique([goal_id, *candidates])

    priorities = (
        "G_MONOTONIC_GLOBAL_ORDER",
        "G_SHARED_GLOBAL_ORDER",
        "G_NONDECREASING_DUPLICATE_ORDER",
        "G_RESTORE_ORDER_AFTER_DEFECT",
        "G_REPAIR_COMPETENCE_PROXY",
        "G_REPAIR_TO_TARGET_MORPHOLOGY",
        "G_HOMEOSTATIC_ORDER_MAINTENANCE",
        "G_NULL_OR_CONTROL_DISTRIBUTION",
        "G_AGGREGATION_TENDENCY",
        "G_DOMINANCE_MOSAIC_INTERFACE_STABILITY",
    )
    for goal_id in priorities:
        if goal_id in candidates:
            return goal_id, "candidate_from_world_priority", candidates
    if candidates:
        return candidates[0], "first_candidate_from_world", candidates
    if valid_goals:
        fallback = sorted(valid_goals)[0]
        return fallback, "fallback_global_goal", [fallback]
    return "", "unresolved_no_goals", []


def _is_metric_column(column: str, value: Any) -> bool:
    if not _not_missing(value):
        return False
    lower = column.lower()
    if any(token in lower for token in CONTEXT_EXCLUDE_TOKENS) and not any(
        token in lower for token in ("score", "metric", "error", "success", "rate", "ratio", "mean", "sd", "sem", "ci", "pvalue", "qvalue")
    ):
        return False
    if isinstance(value, (bool, int, float)) and _float_or_none(value) is not None:
        return any(token in lower for token in METRIC_INCLUDE_TOKENS)
    if isinstance(value, str):
        if len(value) > 160:
            return False
        if lower in {"completed", "stopsreason", "stopreason", "outcomeclassification", "e02verdict", "verdict"}:
            return True
    return False


def _column_plan(columns: Sequence[str]) -> dict[str, list[str]]:
    direct_policy_columns = {
        "abstractpolicyid",
        "sourcepolicyid",
        "policyid",
        "policy_id",
        "policy",
        "policyname",
        "policy_name",
        "policyidsjson",
        "algorithm",
        "algorithms",
        "basepolicy",
        "controllerid",
        "controllerkind",
        "candidateid",
        "sourcecandidateid",
        "genomeid",
        "ablationpolicyid",
        "sourcepolicyids",
        "originalpolicyid",
        "classicpolicyid",
        "discoveredpolicyid",
        "panelpolicyid",
        "leftpanelpolicyid",
        "rightpanelpolicyid",
        "thirdpanelpolicyid",
        "policyapanelpolicyid",
        "policybpanelpolicyid",
        "governanceruleid",
        "interventionpolicyid",
        "basepolicyspecjson",
    }
    plan = {
        "seed": [],
        "scheduler": [],
        "perturbation": [],
        "metric": [],
        "final": [],
        "trajectory": [],
        "raw_trace": [],
        "policy": [],
        "behavior_vector": {},
    }
    normalized_columns = {_norm(column): column for column in map(str, columns)}
    for feature_name, source_names in BEHAVIOR_VECTOR_SOURCES.items():
        for source_name in source_names:
            source_column = normalized_columns.get(_norm(source_name))
            if source_column is not None:
                plan["behavior_vector"][feature_name] = source_column
                break
    for column in map(str, columns):
        lower = column.lower()
        if "seed" in lower:
            plan["seed"].append(column)
        if any(key in lower for key in ("scheduler", "tiebreaker", "activation", "rate", "horizon", "stop")):
            plan["scheduler"].append(column)
        if any(
            key in lower
            for key in (
                "frozen",
                "perturbation",
                "damage",
                "graft",
                "mutant",
                "ratio",
                "arrangement",
                "mixture",
                "shuffle",
                "null",
                "behaviorvariant",
                "behavior_variant",
                "inputprofile",
                "input_profile",
                "goaldirection",
                "goal_direction",
            )
        ):
            plan["perturbation"].append(column)
        if any(token in lower for token in METRIC_INCLUDE_TOKENS) and not (
            any(token in lower for token in CONTEXT_EXCLUDE_TOKENS)
            and not any(
                token in lower
                for token in ("score", "metric", "error", "success", "rate", "ratio", "mean", "sd", "sem", "ci", "pvalue", "qvalue")
            )
        ):
            plan["metric"].append(column)
        if "final" in lower and any(
            token in lower for token in ("state", "values", "positions", "algotypes", "policy", "hash", "frozen", "energy", "error", "success")
        ):
            plan["final"].append(column)
        if any(
            key in lower
            for key in (
                "trajectory",
                "steps",
                "event",
                "activation",
                "swap",
                "comparison",
                "completed",
                "stop",
                "curvature",
                "totalvariation",
                "total_variation",
                "auc",
                "history",
                "trace",
            )
        ):
            plan["trajectory"].append(column)
        if any(token in lower for token in ("path", "artifact", "trace", "run_path")) and "hash" not in lower and "version" not in lower:
            plan["raw_trace"].append(column)
        policy_summary_column = any(
            token in lower
            for token in (
                "arrangementpolicy",
                "initialpolicy",
                "finalpolicy",
                "meanpolicy",
                "policygoal",
                "policycounts",
                "policyassignment",
                "policysatisfied",
                "policyscore",
            )
        )
        policy_id_column = (
            lower in direct_policy_columns
            or lower.endswith("policyid")
            or lower.endswith("policyids")
            or lower.endswith("policyidsjson")
            or lower.endswith("panelpolicyid")
            or lower.endswith("controllerid")
            or lower.endswith("governanceruleid")
        )
        if policy_id_column and not policy_summary_column:
            plan["policy"].append(column)
    return plan


def extract_metric_values(row: Mapping[str, Any], max_items: int = 120, columns: Sequence[str] | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    scan_columns = columns if columns is not None else list(row)
    for column in scan_columns:
        value = row.get(column)
        if len(metrics) >= max_items:
            break
        if _is_metric_column(str(column), value):
            ready = json_ready(value)
            if ready is not None:
                metrics[str(column)] = ready
    return metrics


def _first_present(row: Mapping[str, Any], names: Sequence[str], normalized: Mapping[str, str] | None = None) -> Any:
    normalized = normalized or {_norm(key): key for key in row}
    for name in names:
        key = normalized.get(_norm(name))
        if key is not None and _not_missing(row.get(key)):
            return row.get(key)
    return None


def extract_behavior_vector(row: Mapping[str, Any], vector_plan: Mapping[str, str] | None = None) -> dict[str, Any]:
    vector: dict[str, Any] = {}
    if vector_plan is not None:
        for key, column in vector_plan.items():
            ready = json_ready(row.get(column))
            if ready is not None:
                vector[key] = ready
        return vector
    normalized = {_norm(key): key for key in row}
    for key, names in BEHAVIOR_VECTOR_SOURCES.items():
        value = _first_present(row, names, normalized)
        ready = json_ready(value)
        if ready is not None:
            vector[key] = ready
    return vector


def extract_final_state(row: Mapping[str, Any], max_items: int = 40, columns: Sequence[str] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {}
    scan_columns = columns if columns is not None else list(row)
    for column in scan_columns:
        value = row.get(column)
        ready = json_ready(value)
        if ready is None:
            continue
        state[str(column)] = ready
        if len(state) >= max_items:
            break
    return state


def extract_seed_json(row: Mapping[str, Any], columns: Sequence[str] | None = None) -> dict[str, Any]:
    scan_columns = columns if columns is not None else list(row)
    return {str(column): json_ready(row.get(column)) for column in scan_columns if _not_missing(row.get(column))}


def extract_scheduler_json(row: Mapping[str, Any], columns: Sequence[str] | None = None) -> dict[str, Any]:
    scan_columns = columns if columns is not None else list(row)
    return {str(column): json_ready(row.get(column)) for column in scan_columns if _not_missing(row.get(column))}


def extract_perturbation_json(row: Mapping[str, Any], columns: Sequence[str] | None = None) -> dict[str, Any]:
    scan_columns = columns if columns is not None else list(row)
    return {str(column): json_ready(row.get(column)) for column in scan_columns if _not_missing(row.get(column))}


def extract_trajectory_summary(row: Mapping[str, Any], columns: Sequence[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    scan_columns = columns if columns is not None else list(row)
    for column in scan_columns:
        value = row.get(column)
        if _not_missing(value):
            ready = json_ready(value)
            if ready is not None:
                summary[str(column)] = ready
        if len(summary) >= 80:
            break
    return summary


def _resolve_upstream_path(value: Any, experiment_id: str, previous_artifacts_dir: Path) -> str:
    text = str(value)
    if text.startswith("/previous-artifacts"):
        return text
    if text.startswith("/artifacts/"):
        return str(previous_artifacts_dir / experiment_id / text.removeprefix("/artifacts/"))
    if text.startswith("artifacts/"):
        return str(previous_artifacts_dir / experiment_id / text.removeprefix("artifacts/"))
    if text.startswith("results/") or text.startswith("research_steps/") or text.startswith("traces/"):
        return str(previous_artifacts_dir / experiment_id / text)
    return text


def extract_raw_trace_links(
    row: Mapping[str, Any],
    experiment_id: str,
    previous_artifacts_dir: Path,
    columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    scan_columns = columns if columns is not None else list(row)
    for column in scan_columns:
        value = row.get(column)
        if not _not_missing(value):
            continue
        for item in _as_list(value):
            if not _not_missing(item):
                continue
            resolved = _resolve_upstream_path(item, experiment_id, previous_artifacts_dir)
            links.append(
                {
                    "column": str(column),
                    "value": str(item),
                    "resolvedPath": resolved,
                    "exists": Path(resolved).exists(),
                }
            )
    return links[:20]


def infer_run_id(row: Mapping[str, Any], table_name: str, source_row_index: int) -> str:
    for key in (
        "run_id",
        "runId",
        "sourceRunId",
        "trajectory_id",
        "trajectoryId",
        "conditionId",
        "condition_id",
        "sourceConditionId",
        "analysisId",
        "sourceRunPath",
    ):
        if _not_missing(row.get(key)):
            return str(row[key])
    parts = [table_name, str(source_row_index)]
    for key in ("algorithm", "policyId", "policy_id", "benchmark_id", "task_id", "target_id", "replicateIndex", "seedIndex"):
        if _not_missing(row.get(key)):
            parts.append(str(row[key]))
    return "::".join(parts)


def infer_source_step_id(row: Mapping[str, Any], table_name: str) -> str:
    for key in ("researchStepId", "research_step_id", "sourceStepId", "source_step_id"):
        if _not_missing(row.get(key)):
            return str(row[key])
    match = re.search(r"(?:^|_)s(\d{2})(?:_|$)", table_name.lower())
    if match:
        return f"S{match.group(1)}"
    return "unknown"


def _missingness(record: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "worldIdMissing": not _not_missing(record.get("worldId")),
        "abstractGoalIdMissing": not _not_missing(record.get("abstractGoalId")),
        "abstractPolicyIdsMissing": not bool(record.get("abstractPolicyIdsJson")),
        "metricValuesMissing": not bool(record.get("metricValuesJson")),
        "behaviorVectorMissing": not bool(record.get("behaviorVectorJson")),
        "seedMissing": not bool(record.get("seedJson")),
        "rawTraceLinksMissing": not bool(record.get("rawTraceLinksJson")),
        "finalStateMissing": not bool(record.get("finalStateJson")),
    }


def source_artifact_index(previous_artifacts_dir: str | Path = "/previous-artifacts") -> pd.DataFrame:
    previous_artifacts_dir = Path(previous_artifacts_dir)
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_TABLE_SPECS:
        experiment_id = spec["experimentId"]
        relative_path = spec["relativePath"]
        path = previous_artifacts_dir / experiment_id / relative_path
        table_name = Path(relative_path).stem
        available = path.exists()
        row: dict[str, Any] = {
            "sourceExperimentId": experiment_id,
            "sourceTableName": table_name,
            "sourceKind": spec["sourceKind"],
            "sourceRelativePath": relative_path,
            "sourceTablePath": str(path),
            "available": bool(available),
            "integrationStatus": "available_pending_integration" if available else "missing_source_table",
            "rowCount": None,
            "columnCount": None,
            "sourceTableSha256": None,
            "schemaColumnsJson": [],
            "integratedRowCount": 0,
            "caveatsOrBlockers": [] if available else ["source table was not present at the expected upstream path"],
        }
        if available:
            df = read_table(path)
            row.update(
                {
                    "rowCount": int(len(df)),
                    "columnCount": int(len(df.columns)),
                    "sourceTableSha256": sha256_path(path),
                    "schemaColumnsJson": list(map(str, df.columns)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_world_catalog(path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet") -> pd.DataFrame:
    worlds = read_table(path)
    if worlds.empty:
        raise FileNotFoundError(f"S01 world catalog not found or empty: {path}")
    return worlds


def load_policy_catalog(path: str | Path = "/artifacts/research_steps/S02/policy_abstract_catalog.parquet") -> pd.DataFrame:
    policies = read_table(path)
    if policies.empty:
        raise FileNotFoundError(f"S02 policy catalog not found or empty: {path}")
    return policies


def load_goal_catalog(path: str | Path = "/artifacts/research_steps/S03/goal_catalog.parquet") -> pd.DataFrame:
    goals = read_table(path)
    if goals.empty:
        raise FileNotFoundError(f"S03 goal catalog not found or empty: {path}")
    return goals


def build_behavior_corpus(
    previous_artifacts_dir: str | Path = "/previous-artifacts",
    world_catalog_path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
    policy_catalog_path: str | Path = "/artifacts/research_steps/S02/policy_abstract_catalog.parquet",
    goal_catalog_path: str | Path = "/artifacts/research_steps/S03/goal_catalog.parquet",
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    previous_artifacts_dir = Path(previous_artifacts_dir)
    worlds = load_world_catalog(world_catalog_path)
    policies = load_policy_catalog(policy_catalog_path)
    goals = load_goal_catalog(goal_catalog_path)
    resolver = PolicyResolver(policies)
    world_ids_by_experiment = {
        experiment_id: set(_world_ids(worlds, experiment_id)) for experiment_id in sorted(worlds["experimentId"].astype(str).unique())
    }
    world_to_goals = _parse_goal_world_links(goals)
    goal_policy_lookup = _goal_policy_lookup(goals)
    valid_goal_ids = set(goals["abstractGoalId"].astype(str))
    goal_policy_candidate_cache: dict[tuple[str, ...], list[str]] = {}
    policy_resolution_cache: dict[tuple[Any, ...], tuple[list[str], str, list[str], list[str]]] = {}
    artifact_index = source_artifact_index(previous_artifacts_dir)
    table_hashes = {
        row["sourceTablePath"]: row["sourceTableSha256"]
        for _, row in artifact_index.iterrows()
        if bool(row["available"]) and _not_missing(row["sourceTableSha256"])
    }
    integrated_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for spec in SOURCE_TABLE_SPECS:
        experiment_id = spec["experimentId"]
        relative_path = spec["relativePath"]
        table_path = previous_artifacts_dir / experiment_id / relative_path
        if not table_path.exists():
            continue
        table_name = table_path.stem
        df = read_table(table_path)
        column_plan = _column_plan(list(df.columns))
        table_sha = str(table_hashes.get(str(table_path), sha256_path(table_path)))
        for source_row_index, source_row in enumerate(df.to_dict(orient="records")):
            row = {str(key): json_ready(value) for key, value in source_row.items()}
            world_id, world_status, candidate_worlds = infer_world_id(
                row, experiment_id, table_name, worlds, world_ids_by_experiment
            )
            goal_id, goal_status, candidate_goals = infer_goal_id(
                row, experiment_id, table_name, world_id, goals, world_to_goals, valid_goal_ids
            )
            goal_cache_key = tuple(candidate_goals[:5])
            if goal_cache_key not in goal_policy_candidate_cache:
                goal_policy_candidate_cache[goal_cache_key] = _goal_policy_candidates(goal_policy_lookup, goal_cache_key)
            candidate_goal_policy_ids = goal_policy_candidate_cache[goal_cache_key]
            policy_cache_key = (
                experiment_id,
                world_id,
                tuple((column, str(row.get(column, ""))[:240]) for column in column_plan["policy"]),
            )
            if policy_cache_key not in policy_resolution_cache:
                policy_resolution_cache[policy_cache_key] = resolver.resolve(
                    row, experiment_id, world_id, candidate_goal_policy_ids, column_plan["policy"]
                )
            policy_ids, policy_status, raw_policy_tokens, candidate_policy_ids = policy_resolution_cache[policy_cache_key]
            run_id = infer_run_id(row, table_name, source_row_index)
            row_hash = hashlib.sha256(f"{table_sha}:{source_row_index}".encode("utf-8")).hexdigest()
            behavior_record_id = "BR_" + hashlib.sha256(
                f"{table_path}|{source_row_index}|{row_hash}".encode("utf-8")
            ).hexdigest()[:24]
            record: dict[str, Any] = {
                "behaviorRecordId": behavior_record_id,
                "behaviorSchemaVersion": BEHAVIOR_SCHEMA_VERSION,
                "behaviorCorpusVersion": BEHAVIOR_CORPUS_VERSION,
                "sourceExperimentId": experiment_id,
                "sourceStepId": infer_source_step_id(row, table_name),
                "sourceTableName": table_name,
                "sourceTablePath": str(table_path),
                "sourceTableSha256": table_sha,
                "sourceRowIndex": int(source_row_index),
                "sourceRowHash": row_hash,
                "sourceKind": spec["sourceKind"],
                "worldId": world_id,
                "worldResolutionStatus": world_status,
                "candidateWorldIdsJson": candidate_worlds[:20],
                "abstractGoalId": goal_id,
                "goalResolutionStatus": goal_status,
                "candidateGoalIdsJson": candidate_goals[:30],
                "abstractPolicyIdsJson": policy_ids,
                "policyResolutionStatus": policy_status,
                "policySourceTokensJson": raw_policy_tokens,
                "candidatePolicyIdsJson": candidate_policy_ids,
                "runId": run_id,
                "seedJson": extract_seed_json(row, column_plan["seed"]),
                "schedulerJson": extract_scheduler_json(row, column_plan["scheduler"]),
                "perturbationJson": extract_perturbation_json(row, column_plan["perturbation"]),
                "metricValuesJson": extract_metric_values(row, columns=column_plan["metric"]),
                "behaviorVectorJson": extract_behavior_vector(row, column_plan["behavior_vector"]),
                "finalStateJson": extract_final_state(row, columns=column_plan["final"]),
                "trajectorySummaryJson": extract_trajectory_summary(row, column_plan["trajectory"]),
                "rawTraceLinksJson": extract_raw_trace_links(row, experiment_id, previous_artifacts_dir, column_plan["raw_trace"]),
                "claimBoundary": BEHAVIOR_CLAIM_BOUNDARY,
                "worldSchemaVersion": WORLD_SCHEMA_VERSION,
                "worldCatalogVersion": WORLD_CATALOG_VERSION,
                "policySchemaVersion": POLICY_SCHEMA_VERSION,
                "policyCatalogVersion": POLICY_CATALOG_VERSION,
                "goalSchemaVersion": GOAL_SCHEMA_VERSION,
                "goalCatalogVersion": GOAL_CATALOG_VERSION,
            }
            record["missingnessJson"] = _missingness(record)
            records.append(record)
            integrated_counts[str(table_path)] += 1

    if not artifact_index.empty:
        artifact_index = artifact_index.copy()
        artifact_index["integratedRowCount"] = artifact_index["sourceTablePath"].map(lambda path: int(integrated_counts.get(path, 0)))
        artifact_index["integrationStatus"] = artifact_index.apply(
            lambda row: "integrated" if bool(row["available"]) and int(row["integratedRowCount"]) > 0 else row["integrationStatus"],
            axis=1,
        )
    return records, artifact_index


def corpus_to_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for column in JSON_COLUMNS:
            row[column] = compact_json(row.get(column, [] if column.endswith("IdsJson") else {}))
        if "policySourceTokensJson" in row:
            row["policySourceTokensJson"] = compact_json(row.get("policySourceTokensJson", []))
        rows.append(row)
    return pd.DataFrame(rows)


def artifact_index_to_dataframe(artifact_index: pd.DataFrame) -> pd.DataFrame:
    out = artifact_index.copy()
    for column in ("schemaColumnsJson", "caveatsOrBlockers"):
        if column in out.columns:
            out[column] = out[column].map(compact_json)
    return out


def corpus_dictionary_document() -> dict[str, Any]:
    fields = [
        ("behaviorRecordId", "Stable hash-based row identifier for the unified S04 corpus."),
        ("sourceExperimentId", "Upstream experiment that produced the source row: E01 through E06."),
        ("sourceStepId", "Best available upstream research step ID from the source row or table name."),
        ("sourceTablePath", "Read-only previous-artifact table path used as provenance."),
        ("sourceTableSha256", "SHA-256 checksum for the source table file."),
        ("sourceRowHash", "SHA-256 hash of the normalized upstream row content."),
        ("worldId", "Resolved S01 world catalog identifier."),
        ("abstractGoalId", "Resolved S03 goal catalog identifier."),
        ("abstractPolicyIdsJson", "JSON list of exact S02 abstract policy IDs when row-level policy tokens resolve."),
        ("candidatePolicyIdsJson", "JSON list of candidate S02 policies when row-level policy identity is ambiguous."),
        ("policyResolutionStatus", "Resolution status for policy IDs: exact, candidate, or unresolved."),
        ("seedJson", "JSON object of seed-like fields present in the source row."),
        ("schedulerJson", "JSON object of scheduler, activation, rate, horizon, and stop-condition fields."),
        ("perturbationJson", "JSON object of frozen, null, mixture, arrangement, graft, mutant, or input perturbation fields."),
        ("metricValuesJson", "JSON object of selected numeric, boolean, or categorical behavior metrics from the row."),
        ("behaviorVectorJson", "JSON object of normalized cross-experiment behavior features used by downstream modeling."),
        ("finalStateJson", "JSON object of compact final-state fields and hashes when present."),
        ("trajectorySummaryJson", "JSON object of compact trajectory, event, activation, and completion summaries."),
        ("rawTraceLinksJson", "JSON list of links to raw traces or source artifacts; large upstream traces are not duplicated."),
        ("missingnessJson", "JSON object documenting absent policy, seed, trace, final-state, metric, or behavior-vector fields."),
        ("claimBoundary", "Boundary statement limiting interpretation to computational proxy evidence."),
    ]
    return {
        "researchStepId": "S04",
        "stepNumber": 4,
        "schemaVersion": BEHAVIOR_SCHEMA_VERSION,
        "corpusVersion": BEHAVIOR_CORPUS_VERSION,
        "title": "E07 unified behavior corpus dictionary",
        "completionStatus": "completed when generated with passing validation",
        "artifactsWritten": [
            "unified_behavior_corpus.{parquet,jsonl}",
            "corpus_dictionary.{json,md}",
            "source_artifact_index.{csv,parquet,json}",
            "corpus_validation.{csv,parquet}",
            "missingness_report.{csv,parquet,md}",
            "validation_report.md",
            "summary.md",
            "status.json",
            "artifact_manifest.json",
            "$ARTIFACTS_DIR/data/e07_unified_behavior_corpus.parquet",
        ],
        "validationResult": "see S04 validation report",
        "caveatsOrBlockers": (
            "Heterogeneous upstream schemas are normalized into JSON metric and behavior-vector fields; "
            "large raw trajectories are linked rather than duplicated."
        ),
        "recommendedNextAction": "Chief review, then proceed to S05 behavior predictor training if accepted.",
        "fields": [{"name": name, "description": description} for name, description in fields],
        "jsonColumns": list(JSON_COLUMNS) + ["policySourceTokensJson"],
        "claimBoundary": BEHAVIOR_CLAIM_BOUNDARY,
    }


def validate_behavior_corpus(
    records: Sequence[Mapping[str, Any]],
    artifact_index: pd.DataFrame,
    world_catalog_path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
    policy_catalog_path: str | Path = "/artifacts/research_steps/S02/policy_abstract_catalog.parquet",
    goal_catalog_path: str | Path = "/artifacts/research_steps/S03/goal_catalog.parquet",
) -> pd.DataFrame:
    worlds = load_world_catalog(world_catalog_path)
    policies = load_policy_catalog(policy_catalog_path)
    goals = load_goal_catalog(goal_catalog_path)
    valid_worlds = set(worlds["worldId"].astype(str))
    valid_policies = set(policies["abstractPolicyId"].astype(str))
    valid_goals = set(goals["abstractGoalId"].astype(str))
    table_hashes = {
        str(row["sourceTablePath"]): str(row["sourceTableSha256"])
        for _, row in artifact_index.iterrows()
        if bool(row.get("available")) and _not_missing(row.get("sourceTableSha256"))
    }
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, success: bool, severity: str = "error", detail: str = "") -> None:
        checks.append({"scope": scope, "check": check, "success": bool(success), "severity": severity, "detail": detail})

    add("corpus", "row_count_positive", len(records) > 0, detail=str(len(records)))
    represented = {str(record.get("sourceExperimentId")) for record in records}
    add(
        "corpus",
        "all_source_experiments_represented",
        represented == VALIDATION_REQUIRED_EXPERIMENTS,
        detail=compact_json(sorted(represented)),
    )
    integrated_tables = set(str(record.get("sourceTablePath")) for record in records)
    available_tables = {str(row["sourceTablePath"]) for _, row in artifact_index.iterrows() if bool(row.get("available"))}
    add(
        "artifact_index",
        "all_available_source_tables_integrated",
        available_tables.issubset(integrated_tables),
        detail=f"{len(integrated_tables)}/{len(available_tables)}",
    )

    for _, row in artifact_index.iterrows():
        scope = str(row["sourceTableName"])
        available = bool(row.get("available"))
        add(scope, "source_table_available", available, "warning" if not available else "error", str(row.get("sourceTablePath")))
        if available:
            add(scope, "source_table_checksum_recorded", _not_missing(row.get("sourceTableSha256")), detail=str(row.get("sourceTablePath")))
            add(scope, "source_table_integrated_rows_match", int(row.get("rowCount") or 0) == int(row.get("integratedRowCount") or 0), detail=f"{row.get('integratedRowCount')}/{row.get('rowCount')}")

    record_ids = [str(record.get("behaviorRecordId")) for record in records]
    add("corpus", "behavior_record_ids_unique", len(record_ids) == len(set(record_ids)), detail=str(len(record_ids)))

    by_table: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_table[str(record.get("sourceTableName"))].append(record)

    nullable_required = {
        "abstractPolicyIdsJson",
        "candidatePolicyIdsJson",
        "seedJson",
        "schedulerJson",
        "perturbationJson",
        "metricValuesJson",
        "behaviorVectorJson",
        "finalStateJson",
        "trajectorySummaryJson",
        "rawTraceLinksJson",
    }
    for table_name, table_records in sorted(by_table.items()):
        scope = f"table::{table_name}"
        for field in REQUIRED_BEHAVIOR_FIELDS:
            missing = [
                str(record.get("behaviorRecordId"))
                for record in table_records
                if field not in record or (field not in nullable_required and not _not_missing(record.get(field)))
            ]
            add(
                scope,
                f"required_field_populated::{field}",
                not missing,
                detail=compact_json({"missingCount": len(missing), "examples": missing[:5]}),
            )

        unresolved_worlds = [str(record.get("worldId")) for record in table_records if str(record.get("worldId")) not in valid_worlds]
        unresolved_goals = [str(record.get("abstractGoalId")) for record in table_records if str(record.get("abstractGoalId")) not in valid_goals]
        unresolved_policy_ids = [
            str(policy_id)
            for record in table_records
            for policy_id in record.get("abstractPolicyIdsJson", [])
            if str(policy_id) not in valid_policies
        ]
        missing_policy_rows = [record for record in table_records if not record.get("abstractPolicyIdsJson")]
        missing_metrics = [
            str(record.get("behaviorRecordId"))
            for record in table_records
            if not record.get("metricValuesJson") and not record.get("behaviorVectorJson")
        ]
        checksum_mismatches = [
            str(record.get("behaviorRecordId"))
            for record in table_records
            if str(record.get("sourceTableSha256")) != table_hashes.get(str(record.get("sourceTablePath")))
        ]
        undocumented_policy = [
            str(record.get("behaviorRecordId"))
            for record in table_records
            if str(record.get("policyResolutionStatus"))
            not in {"exact_policy_id_or_source_token", "candidate_from_world_or_goal", "unresolved_no_policy_token"}
        ]

        add(scope, "all_worlds_resolve_to_s01", not unresolved_worlds, detail=compact_json(Counter(unresolved_worlds)))
        add(scope, "all_goals_resolve_to_s03", not unresolved_goals, detail=compact_json(Counter(unresolved_goals)))
        add(scope, "policy_ids_resolve_when_present", not unresolved_policy_ids, detail=compact_json(Counter(unresolved_policy_ids)))
        add(scope, "policy_resolution_documented", not undocumented_policy, detail=compact_json(undocumented_policy[:5]))
        add(
            scope,
            "row_level_policy_exact_link_present",
            not missing_policy_rows,
            "warning",
            detail=f"{len(table_records) - len(missing_policy_rows)}/{len(table_records)} exact row-level links",
        )
        add(scope, "metric_or_behavior_vector_present", not missing_metrics, "warning", detail=compact_json(missing_metrics[:5]))
        add(scope, "source_table_checksum_matches_index", not checksum_mismatches, detail=compact_json(checksum_mismatches[:5]))

    return pd.DataFrame(checks)


def missingness_report(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not records:
        return pd.DataFrame(columns=["field", "missingCount", "rowCount", "missingFraction", "sourceExperimentBreakdownJson"])
    missing_fields = (
        "abstractPolicyIdsMissing",
        "metricValuesMissing",
        "behaviorVectorMissing",
        "seedMissing",
        "rawTraceLinksMissing",
        "finalStateMissing",
    )
    row_count = len(records)
    for field in missing_fields:
        missing_records = [record for record in records if bool(record.get("missingnessJson", {}).get(field))]
        by_experiment = Counter(str(record.get("sourceExperimentId")) for record in missing_records)
        rows.append(
            {
                "field": field,
                "missingCount": len(missing_records),
                "rowCount": row_count,
                "missingFraction": len(missing_records) / row_count,
                "sourceExperimentBreakdownJson": dict(sorted(by_experiment.items())),
            }
        )
    return pd.DataFrame(rows)


def coverage_summary(records: Sequence[Mapping[str, Any]], validation: pd.DataFrame, artifact_index: pd.DataFrame) -> dict[str, Any]:
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])] if not validation.empty else pd.DataFrame()
    warnings = validation[(validation["severity"].eq("warning")) & (~validation["success"])] if not validation.empty else pd.DataFrame()
    source_counts = Counter(str(record.get("sourceExperimentId")) for record in records)
    table_counts = Counter(str(record.get("sourceTableName")) for record in records)
    world_counts = Counter(str(record.get("worldId")) for record in records)
    goal_counts = Counter(str(record.get("abstractGoalId")) for record in records)
    policy_status_counts = Counter(str(record.get("policyResolutionStatus")) for record in records)
    exact_policy_rows = sum(1 for record in records if record.get("abstractPolicyIdsJson"))
    linked_policy_ids = {
        str(policy_id)
        for record in records
        for policy_id in record.get("abstractPolicyIdsJson", [])
        if _not_missing(policy_id)
    }
    return {
        "behaviorSchemaVersion": BEHAVIOR_SCHEMA_VERSION,
        "behaviorCorpusVersion": BEHAVIOR_CORPUS_VERSION,
        "recordCount": len(records),
        "sourceTableCount": int(len(artifact_index)),
        "availableSourceTableCount": int(artifact_index["available"].sum()) if "available" in artifact_index else 0,
        "integratedSourceTableCount": len({str(record.get("sourceTableName")) for record in records}),
        "sourceExperimentCounts": dict(sorted(source_counts.items())),
        "sourceTableCounts": dict(sorted(table_counts.items())),
        "worldCount": len(world_counts),
        "goalCount": len(goal_counts),
        "exactPolicyLinkedRowCount": exact_policy_rows,
        "exactPolicyLinkedRowFraction": exact_policy_rows / len(records) if records else 0.0,
        "linkedExactPolicyCount": len(linked_policy_ids),
        "policyResolutionStatusCounts": dict(sorted(policy_status_counts.items())),
        "validationCheckCount": int(len(validation)),
        "hardValidationFailureCount": int(len(hard_failures)),
        "warningFailureCount": int(len(warnings)),
        "allHardChecksPassed": bool(hard_failures.empty),
        "claimBoundary": BEHAVIOR_CLAIM_BOUNDARY,
    }
