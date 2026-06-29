from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "e07_s01_world_schema.v1"
CATALOG_VERSION = "e07_s01_world_catalog.v1"
CLAIM_BOUNDARY = (
    "Computational local-policy world schema only; not direct evidence about biological morphogenesis, "
    "cognition, agency, sentience, or living chimeras."
)
TUPLE_FIELDS = (
    "stateSpace",
    "localObservations",
    "actionSet",
    "transitionRules",
    "goalPredicateOrEnergy",
    "perturbationModel",
    "scheduler",
    "measurementFunctions",
)
JSON_COLUMNS = (
    "sourceStepIds",
    "stateSpace",
    "localObservations",
    "actionSet",
    "transitionRules",
    "goalPredicateOrEnergy",
    "perturbationModel",
    "scheduler",
    "measurementFunctions",
    "upstreamArtifactPaths",
    "upstreamStatusPaths",
    "migrationHints",
    "exceptionsOrGaps",
)
REQUIRED_RECORD_FIELDS = (
    "worldId",
    "schemaVersion",
    "catalogVersion",
    "experimentId",
    "sourceStepIds",
    "title",
    "worldFamily",
    "substrateClass",
    *TUPLE_FIELDS,
    "upstreamArtifactPaths",
    "upstreamStatusPaths",
    "replayability",
    "metadataCompleteness",
    "claimBoundary",
)


def json_ready(value: Any) -> Any:
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_schema_document() -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "title": "E07 formal world tuple schema",
        "claimBoundary": CLAIM_BOUNDARY,
        "formalTuple": [
            {
                "field": "stateSpace",
                "description": "Allowed substrate, cells or agents, identity/state variables, conserved quantities, and dimensionality.",
            },
            {
                "field": "localObservations",
                "description": "Information available to a policy at an activation, including locality radius, neighbor fields, memory, signal fields, and excluded global information.",
            },
            {
                "field": "actionSet",
                "description": "Primitive local operations such as wait, adjacent swap, target swap, emit signal, nudge, insert/delete, divide, graft, or governance action.",
            },
            {
                "field": "transitionRules",
                "description": "Update semantics for state, identities, memory, signals, perturbations, frozen or repair states, and action legality.",
            },
            {
                "field": "goalPredicateOrEnergy",
                "description": "Success predicate or energy/proxy minimized by the world, with target compatibility and directionality metadata.",
            },
            {
                "field": "perturbationModel",
                "description": "Input distribution, freezing, scheduler/null manipulation, damage, graft, mutant, history, or morphology perturbations.",
            },
            {
                "field": "scheduler",
                "description": "Activation-order family, stochastic seeds, deterministic variants, rate control, stopping rules, and caps.",
            },
            {
                "field": "measurementFunctions",
                "description": "Metrics emitted by the world, including direct simulator measurements and bounded computational proxies.",
            },
        ],
        "worldRecordSchema": {
            "type": "object",
            "required": list(REQUIRED_RECORD_FIELDS),
            "properties": {
                "worldId": {"type": "string", "description": "Stable E07 world identifier."},
                "schemaVersion": {"const": SCHEMA_VERSION},
                "catalogVersion": {"const": CATALOG_VERSION},
                "experimentId": {"type": "string", "enum": ["E01", "E02", "E03", "E04", "E05", "E06"]},
                "sourceStepIds": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"},
                "worldFamily": {
                    "type": "string",
                    "description": "High-level family such as sorting, scheduler_control, null_model, morphospace, repair, morphology, or chimera.",
                },
                "substrateClass": {
                    "type": "string",
                    "description": "Substrate such as row_1d, square_grid_2d, graph, mixed_1d_chimera, or abstract_metric_panel.",
                },
                "stateSpace": {"type": "object"},
                "localObservations": {"type": "object"},
                "actionSet": {"type": "object"},
                "transitionRules": {"type": "object"},
                "goalPredicateOrEnergy": {"type": "object"},
                "perturbationModel": {"type": "object"},
                "scheduler": {"type": "object"},
                "measurementFunctions": {"type": "array", "items": {"type": "object"}},
                "upstreamArtifactPaths": {"type": "array", "items": {"type": "string"}},
                "upstreamStatusPaths": {"type": "array", "items": {"type": "string"}},
                "replayability": {"type": "string", "enum": ["direct", "derived", "partial", "summary_only"]},
                "metadataCompleteness": {"type": "string", "enum": ["complete", "partial", "exception_documented"]},
                "exceptionsOrGaps": {"type": "array", "items": {"type": "string"}},
                "migrationHints": {"type": "object"},
                "sourceRecordCount": {"type": ["integer", "null"]},
                "notes": {"type": "string"},
                "claimBoundary": {"type": "string"},
            },
        },
        "normalizationContract": {
            "recordIdentity": ["worldId", "experimentId", "sourceStepIds"],
            "requiredLinks": ["upstreamArtifactPaths", "upstreamStatusPaths"],
            "validationChecks": [
                "required record fields are populated",
                "formal tuple fields have the expected JSON shape",
                "linked upstream artifacts exist",
                "linked upstream statuses exist and report success",
                "partial and summary-only records document exceptions or gaps",
            ],
        },
    }


def _root(previous_artifacts_dir: Path, experiment_id: str) -> Path:
    return previous_artifacts_dir / experiment_id


def _status(previous_artifacts_dir: Path, experiment_id: str, step_id: str) -> str:
    return str(_root(previous_artifacts_dir, experiment_id) / "research_steps" / step_id / "status.json")


def _artifact(previous_artifacts_dir: Path, experiment_id: str, *parts: str) -> str:
    return str(_root(previous_artifacts_dir, experiment_id).joinpath(*parts))


def _existing(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in paths:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _row_count(path: str | Path, filters: Mapping[str, Any] | None = None) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    if filters:
        for column, expected in filters.items():
            if column not in df.columns:
                return 0
            if isinstance(expected, (set, list, tuple)):
                df = df[df[column].isin(list(expected))]
            else:
                df = df[df[column] == expected]
    return int(len(df))


def _unique_values(path: str | Path, column: str, limit: int = 50) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False) if path.suffix != ".parquet" else pd.read_parquet(path)
    if column not in df.columns:
        return []
    values = [str(value) for value in df[column].dropna().unique()]
    return sorted(values)[:limit]


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False) if path.suffix != ".parquet" else pd.read_parquet(path)
    return df.to_dict(orient="records")


def _base_state(substrate: str = "row_1d", n: Any = "varies", identities: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "substrate": substrate,
        "size": n,
        "agents": "one policy-bearing cell per occupied node",
        "identityFields": list(identities or ["scalar_value", "algotype", "optional_memory"]),
        "conservedQuantities": ["cell_count"] if substrate != "nonconservative_morphology" else ["lineage_ids_when_present"],
    }


def _row_observations(extra: Sequence[str] | None = None, radius: int = 1) -> dict[str, Any]:
    fields = ["own_value", "own_algotype", "left_neighbor", "right_neighbor", "frozen_or_blocked_state"]
    fields.extend(extra or [])
    return {
        "locality": f"radius_{radius}",
        "fields": sorted(set(fields)),
        "globalInformationExcluded": True,
    }


def _row_actions(extra: Sequence[str] | None = None) -> dict[str, Any]:
    actions = ["wait", "swap_left", "swap_right", "selection_target_swap_when_policy_allows"]
    actions.extend(extra or [])
    return {"primitiveActions": sorted(set(actions)), "legality": "local adjacency or declared target/action guard"}


def _sorting_goal(direction: str = "increasing", goal_type: str = "strict_global_order") -> dict[str, Any]:
    return {
        "goalType": goal_type,
        "direction": direction,
        "predicate": "final array satisfies declared monotonic order or source-specific target-quality proxy",
        "energy": "monotonicity_error, inversion distance, or target-quality error",
    }


def _sorting_measurements(extra: Sequence[str] | None = None) -> list[dict[str, Any]]:
    names = [
        "Sortedness",
        "monotonicity_error",
        "swap_count",
        "comparison_count",
        "activation_count",
        "Delayed Gratification",
        "Aggregation",
        "Kendall tau distance",
        "Earth-mover position distance",
    ]
    names.extend(extra or [])
    return [{"name": name, "type": "computational_proxy" if name in {"Delayed Gratification", "Aggregation"} else "simulator_measurement"} for name in names]


def _record(
    *,
    world_id: str,
    experiment_id: str,
    source_steps: Sequence[str],
    title: str,
    world_family: str,
    substrate_class: str,
    state_space: Mapping[str, Any],
    local_observations: Mapping[str, Any],
    action_set: Mapping[str, Any],
    transition_rules: Mapping[str, Any],
    goal: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    measurement_functions: Sequence[Mapping[str, Any]],
    artifact_paths: Sequence[str],
    status_paths: Sequence[str],
    replayability: str,
    metadata_completeness: str,
    source_record_count: int | None = None,
    exceptions_or_gaps: Sequence[str] | None = None,
    migration_hints: Mapping[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "worldId": world_id,
        "schemaVersion": SCHEMA_VERSION,
        "catalogVersion": CATALOG_VERSION,
        "experimentId": experiment_id,
        "sourceStepIds": list(source_steps),
        "title": title,
        "worldFamily": world_family,
        "substrateClass": substrate_class,
        "stateSpace": dict(state_space),
        "localObservations": dict(local_observations),
        "actionSet": dict(action_set),
        "transitionRules": dict(transition_rules),
        "goalPredicateOrEnergy": dict(goal),
        "perturbationModel": dict(perturbation),
        "scheduler": dict(scheduler),
        "measurementFunctions": [dict(item) for item in measurement_functions],
        "upstreamArtifactPaths": _existing(artifact_paths),
        "upstreamStatusPaths": _existing(status_paths),
        "replayability": replayability,
        "metadataCompleteness": metadata_completeness,
        "exceptionsOrGaps": list(exceptions_or_gaps or []),
        "migrationHints": dict(migration_hints or {}),
        "sourceRecordCount": source_record_count,
        "notes": notes,
        "claimBoundary": CLAIM_BOUNDARY,
    }


def _e01_records(previous: Path) -> list[dict[str, Any]]:
    cond = _artifact(previous, "E01", "research_steps", "S03", "condition_matrix.csv")
    seed = _artifact(previous, "E01", "research_steps", "S03", "seed_table.csv")
    return [
        _record(
            world_id="E01_S04_1d_sorting_baseline",
            experiment_id="E01",
            source_steps=["S03", "S04", "S05", "S06"],
            title="Faithful 1D sorting baseline",
            world_family="sorting",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100),
            local_observations=_row_observations(),
            action_set=_row_actions(),
            transition_rules={"family": "original_and_deterministic_cell_view", "update": "asynchronous local compare/swap until sorted_or_cap"},
            goal=_sorting_goal("increasing"),
            perturbation={"inputProfile": "unique_1_100", "frozenCells": 0, "chimera": False},
            scheduler={"family": "asynchronous_random_cell_activation", "seeds": ["schedulerSeed", "tieBreakerSeed"], "stopPolicy": "sorted_or_cap"},
            measurement_functions=_sorting_measurements(["efficiency_z_tests", "bootstrap_intervals"]),
            artifact_paths=[cond, seed, _artifact(previous, "E01", "results", "e01_results.parquet")],
            status_paths=[_status(previous, "E01", step) for step in ["S03", "S04", "S05", "S06"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(cond, {"producerStep": "S04"}),
            migration_hints={"primaryRunKeys": ["conditionId", "replicateIndex", "schedulerSeed", "tieBreakerSeed"]},
        ),
        _record(
            world_id="E01_S07_frozen_cell_sorting",
            experiment_id="E01",
            source_steps=["S07", "S08"],
            title="1D sorting with Frozen Cell perturbations",
            world_family="repair_robustness",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100, ["scalar_value", "algotype", "frozen_state"]),
            local_observations=_row_observations(["neighbor_frozen_state"]),
            action_set=_row_actions(["blocked_attempt"]),
            transition_rules={"family": "frozen_cell_variants", "frozenVariants": ["passive", "stuck"], "update": "swap blocked or allowed according to frozen behavior"},
            goal=_sorting_goal("increasing"),
            perturbation={"frozenCells": "1_or_more", "placementSeed": "frozenPositionSeed", "inputProfile": "unique_1_100"},
            scheduler={"family": "asynchronous_random_cell_activation", "stopPolicy": "sorted_or_cap"},
            measurement_functions=_sorting_measurements(["frozen_swap_attempt_rate", "robustness_score"]),
            artifact_paths=[cond, seed, _artifact(previous, "E01", "results", "e01_frozen_cell_robustness.parquet")],
            status_paths=[_status(previous, "E01", step) for step in ["S07", "S08"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(cond, {"producerStep": "S07"}),
            migration_hints={"perturbationFields": ["frozenVariant", "frozenCount", "frozenPositionSeed"]},
        ),
        _record(
            world_id="E01_S09_same_goal_chimera_sorting",
            experiment_id="E01",
            source_steps=["S09", "S10"],
            title="Same-goal mixed-Algotype 1D chimeras",
            world_family="chimera",
            substrate_class="mixed_1d_chimera",
            state_space=_base_state("row_1d", 100, ["scalar_value", "algotype", "goal_direction"]),
            local_observations=_row_observations(["neighbor_algotype"]),
            action_set=_row_actions(),
            transition_rules={"family": "mixed_policy_cell_view", "update": "each cell executes its assigned local sorting policy"},
            goal=_sorting_goal("increasing", "shared_global_order"),
            perturbation={"mixture": "multiple classic algotypes", "goalCompatibility": "same_goal", "inputProfile": "unique_1_100"},
            scheduler={"family": "asynchronous_random_cell_activation", "assignmentSeed": "algotypeAssignmentSeed"},
            measurement_functions=_sorting_measurements(["per_algotype_movement", "aggregation_peak"]),
            artifact_paths=[cond, seed, _artifact(previous, "E01", "results", "e01_same_goal_chimeras.parquet")],
            status_paths=[_status(previous, "E01", step) for step in ["S09", "S10"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(cond, {"producerStep": "S09"}),
            migration_hints={"chimeraFields": ["algotypeAllocation", "goalDirections", "mixtureId"]},
        ),
        _record(
            world_id="E01_S11_duplicate_value_chimera_sorting",
            experiment_id="E01",
            source_steps=["S11"],
            title="Duplicate-value same-goal 1D chimeras",
            world_family="chimera",
            substrate_class="mixed_1d_chimera",
            state_space=_base_state("row_1d", 100, ["scalar_value_with_duplicates", "algotype", "goal_direction"]),
            local_observations=_row_observations(["equal_value_tie_state", "neighbor_algotype"]),
            action_set=_row_actions(),
            transition_rules={"family": "duplicate_value_mixed_policy", "tieHandling": "source implementation tie rules"},
            goal=_sorting_goal("increasing", "nondecreasing_order"),
            perturbation={"inputProfile": "duplicate_values", "goalCompatibility": "same_goal"},
            scheduler={"family": "asynchronous_random_cell_activation", "stopPolicy": "sorted_or_cap"},
            measurement_functions=_sorting_measurements(["duplicate_value_aggregation"]),
            artifact_paths=[cond, seed, _artifact(previous, "E01", "results", "e01_duplicate_value_chimeras.parquet")],
            status_paths=[_status(previous, "E01", "S11")],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(cond, {"producerStep": "S11"}),
        ),
        _record(
            world_id="E01_S12_opposite_direction_chimera_sorting",
            experiment_id="E01",
            source_steps=["S12"],
            title="Opposite-direction 1D chimeras",
            world_family="chimera_conflict",
            substrate_class="mixed_1d_chimera",
            state_space=_base_state("row_1d", 100, ["scalar_value", "algotype", "per_cell_goal_direction"]),
            local_observations=_row_observations(["neighbor_algotype", "own_goal_direction"]),
            action_set=_row_actions(),
            transition_rules={"family": "opposed_mixed_policy", "update": "cells can pursue increasing or decreasing direction"},
            goal=_sorting_goal("mixed", "conflicting_global_order"),
            perturbation={"goalCompatibility": "opposite_goal", "goalDirections": ["increasing", "decreasing"]},
            scheduler={"family": "asynchronous_random_cell_activation", "stopPolicy": "equilibrium_or_cap"},
            measurement_functions=_sorting_measurements(["equilibrium_class", "dominance_proxy"]),
            artifact_paths=[cond, seed, _artifact(previous, "E01", "results", "e01_opposite_direction_chimeras.parquet")],
            status_paths=[_status(previous, "E01", "S12")],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(cond, {"producerStep": "S12"}),
        ),
    ]


def _e02_records(previous: Path) -> list[dict[str, Any]]:
    return [
        _record(
            world_id="E02_S01_deterministic_reference_sorting",
            experiment_id="E02",
            source_steps=["S01"],
            title="Deterministic reference simulator for E01 worlds",
            world_family="sorting_reference",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", "varies"),
            local_observations=_row_observations(),
            action_set=_row_actions(),
            transition_rules={"family": "deterministic_event_simulator", "compatibilityTarget": "E01 classic cell-view traces"},
            goal=_sorting_goal("declared_by_condition"),
            perturbation={"inheritsFrom": "E01 condition matrix", "variants": ["none", "frozen", "chimera"]},
            scheduler={"family": "seeded_deterministic_event", "replayKeys": ["scheduler_seed", "tie_breaker_seed"]},
            measurement_functions=_sorting_measurements(["trace_state_hash"]),
            artifact_paths=[
                _artifact(previous, "E02", "research_steps", "S01", "trace_schema.md"),
                _artifact(previous, "E02", "research_steps", "S01", "e01_vs_e02_validation.parquet"),
            ],
            status_paths=[_status(previous, "E02", "S01")],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E02", "research_steps", "S01", "e01_vs_e02_validation.parquet")),
        ),
        _record(
            world_id="E02_S02_scheduler_regime_worlds",
            experiment_id="E02",
            source_steps=["S02"],
            title="Scheduler-regime control worlds",
            world_family="scheduler_control",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100),
            local_observations=_row_observations(),
            action_set=_row_actions(),
            transition_rules={"family": "same_policy_different_activation_order", "invariant": "policy code unchanged"},
            goal=_sorting_goal("increasing"),
            perturbation={"schedulerRegimes": _unique_values(_artifact(previous, "E02", "results", "e02_scheduler_regime_summary.csv"), "schedulerRegime")},
            scheduler={"family": "random_round_robin_adversarial_and_rate_variants", "stopPolicy": "sorted_or_cap"},
            measurement_functions=_sorting_measurements(["scheduler_rounds", "activation_share"]),
            artifact_paths=[_artifact(previous, "E02", "results", "e02_scheduler_regime_summary.parquet")],
            status_paths=[_status(previous, "E02", "S02")],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E02", "results", "e02_scheduler_regime_summary.csv")),
        ),
        _record(
            world_id="E02_S03_activation_rate_worlds",
            experiment_id="E02",
            source_steps=["S03"],
            title="Activation-rate manipulation worlds",
            world_family="scheduler_control",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100),
            local_observations=_row_observations(),
            action_set=_row_actions(),
            transition_rules={"family": "same_policy_rate_weighted_activation", "invariant": "local transition rules unchanged"},
            goal=_sorting_goal("increasing"),
            perturbation={"activationRates": "target distributions vary by policy or cell class"},
            scheduler={"family": "rate_weighted_seeded_activation", "control": "realized activation shares checked"},
            measurement_functions=_sorting_measurements(["activation_rate_realization"]),
            artifact_paths=[_artifact(previous, "E02", "results", "e02_activation_rate_summary.parquet")],
            status_paths=[_status(previous, "E02", "S03")],
            replayability="derived",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E02", "results", "e02_activation_rate_summary.csv")),
        ),
        _record(
            world_id="E02_S04_S07_aggregation_null_worlds",
            experiment_id="E02",
            source_steps=["S04", "S05", "S06", "S07"],
            title="Aggregation null-model worlds",
            world_family="null_model",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100, ["scalar_value", "display_label_or_dummy_algotype"]),
            local_observations=_row_observations(["label_or_dummy_policy_assignment"]),
            action_set=_row_actions(["random_local_swap_when_null_allows"]),
            transition_rules={"family": "label_shuffle_dummy_speed_matched_and_local_move_nulls", "policyCode": "varies by null model"},
            goal={"goalType": "control_distribution", "predicate": "matched labels, activation shares, or swap counts rather than biological target"},
            perturbation={"nullModels": ["trajectory_preserving_label_shuffle", "dummy_algotype", "speed_matched_algotype", "local_move_null"]},
            scheduler={"family": "matched_or_seeded_activation", "matchingTargets": ["activation_history", "swap_count", "speed_profile"]},
            measurement_functions=_sorting_measurements(["null_distribution", "observed_minus_null_effect"]),
            artifact_paths=[
                _artifact(previous, "E02", "results", "e02_label_shuffle_mixture_summary.parquet"),
                _artifact(previous, "E02", "results", "e02_dummy_algotypes_summary.parquet"),
                _artifact(previous, "E02", "results", "e02_speed_matched_summary.parquet"),
                _artifact(previous, "E02", "results", "e02_local_move_null_summary.parquet"),
            ],
            status_paths=[_status(previous, "E02", step) for step in ["S04", "S05", "S06", "S07"]],
            replayability="derived",
            metadata_completeness="complete",
            source_record_count=sum(
                value or 0
                for value in [
                    _row_count(_artifact(previous, "E02", "results", "e02_label_shuffle_mixture_summary.csv")),
                    _row_count(_artifact(previous, "E02", "results", "e02_dummy_algotypes_summary.csv")),
                    _row_count(_artifact(previous, "E02", "results", "e02_speed_matched_summary.csv")),
                    _row_count(_artifact(previous, "E02", "results", "e02_local_move_null_summary.csv")),
                ]
            ),
        ),
        _record(
            world_id="E02_S08_delayed_gratification_null_worlds",
            experiment_id="E02",
            source_steps=["S08"],
            title="Delayed-Gratification trajectory null worlds",
            world_family="null_model",
            substrate_class="trajectory_panel",
            state_space={"substrate": "trajectory_summary", "elements": "matched start/end sortedness and swap-count trajectories"},
            local_observations={"locality": "not_policy_execution", "fields": ["trajectory_sortedness", "matched_noise"], "globalInformationExcluded": False},
            action_set={"primitiveActions": ["trajectory_delta_shuffle"], "legality": "preserve declared matching constraints"},
            transition_rules={"family": "trajectory_resampling_null", "update": "constructs null sortedness paths, not direct cell movement"},
            goal={"goalType": "DG_null_distribution", "predicate": "observed DG compared with matched null"},
            perturbation={"nullModel": _unique_values(_artifact(previous, "E02", "results", "e02_dg_null_summary.csv"), "nullModel")},
            scheduler={"family": "matched_trajectory_resampling", "notDirectSimulatorScheduler": True},
            measurement_functions=[{"name": "Delayed Gratification null contrast", "type": "computational_proxy"}],
            artifact_paths=[_artifact(previous, "E02", "results", "e02_dg_null_summary.parquet")],
            status_paths=[_status(previous, "E02", "S08")],
            replayability="derived",
            metadata_completeness="partial",
            source_record_count=_row_count(_artifact(previous, "E02", "results", "e02_dg_null_summary.csv")),
            exceptions_or_gaps=["This world is a trajectory-level null, not a complete local-policy transition world."],
        ),
        _record(
            world_id="E02_S09_S10_metric_distribution_worlds",
            experiment_id="E02",
            source_steps=["S09", "S10"],
            title="Alternative metric and input-distribution worlds",
            world_family="metric_distribution_control",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100),
            local_observations=_row_observations(),
            action_set=_row_actions(),
            transition_rules={"family": "classic_policy_with_metric_or_input_variation"},
            goal=_sorting_goal("increasing"),
            perturbation={"inputProfiles": _unique_values(_artifact(previous, "E02", "results", "e02_input_distribution_summary.csv"), "inputProfile"), "metrics": "alternative sortedness and distance metrics"},
            scheduler={"family": "seeded_activation", "stopPolicy": "sorted_or_cap"},
            measurement_functions=_sorting_measurements(["alternative_metric_panel"]),
            artifact_paths=[
                _artifact(previous, "E02", "results", "e02_alternative_metric_summary.parquet"),
                _artifact(previous, "E02", "results", "e02_input_distribution_summary.parquet"),
            ],
            status_paths=[_status(previous, "E02", step) for step in ["S09", "S10"]],
            replayability="derived",
            metadata_completeness="complete",
            source_record_count=sum(
                value or 0
                for value in [
                    _row_count(_artifact(previous, "E02", "results", "e02_alternative_metric_summary.csv")),
                    _row_count(_artifact(previous, "E02", "results", "e02_input_distribution_summary.csv")),
                ]
            ),
        ),
        _record(
            world_id="E02_S11_S13_frozen_and_stop_variant_worlds",
            experiment_id="E02",
            source_steps=["S11", "S12", "S13", "S14"],
            title="Frozen placement, Frozen behavior, stop-condition, and statistics variants",
            world_family="robustness_control",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100, ["scalar_value", "algotype", "frozen_state"]),
            local_observations=_row_observations(["neighbor_frozen_state"]),
            action_set=_row_actions(["blocked_attempt"]),
            transition_rules={"family": "frozen_placement_behavior_and_stop_condition_variants"},
            goal=_sorting_goal("increasing"),
            perturbation={"variantFamilies": ["frozen_placement", "frozen_behavior", "stop_condition", "strong_statistics"]},
            scheduler={"family": "seeded_activation", "stopPolicy": "variant_specific"},
            measurement_functions=_sorting_measurements(["frozen_variant_effect", "fdr_adjusted_claim_status"]),
            artifact_paths=[
                _artifact(previous, "E02", "results", "e02_frozen_placement_summary.parquet"),
                _artifact(previous, "E02", "results", "e02_frozen_behavior_variant_summary.parquet"),
                _artifact(previous, "E02", "results", "e02_stop_condition_summary.parquet"),
                _artifact(previous, "E02", "results", "e02_strong_statistics_claim_summary.parquet"),
            ],
            status_paths=[_status(previous, "E02", step) for step in ["S11", "S12", "S13", "S14"]],
            replayability="derived",
            metadata_completeness="complete",
            source_record_count=sum(
                value or 0
                for value in [
                    _row_count(_artifact(previous, "E02", "results", "e02_frozen_placement_summary.csv")),
                    _row_count(_artifact(previous, "E02", "results", "e02_frozen_behavior_variant_summary.csv")),
                    _row_count(_artifact(previous, "E02", "results", "e02_stop_condition_summary.csv")),
                    _row_count(_artifact(previous, "E02", "results", "e02_strong_statistics_claim_summary.csv")),
                ]
            ),
            exceptions_or_gaps=["S14 is a claim-statistics layer over prior worlds, so it is represented as measurement metadata rather than a new simulator substrate."],
        ),
    ]


def _e03_records(previous: Path) -> list[dict[str, Any]]:
    return [
        _record(
            world_id="E03_S01_policy_interface_replay_world",
            experiment_id="E03",
            source_steps=["S01"],
            title="Policy-interface replay world",
            world_family="policy_morphospace",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", "varies"),
            local_observations=_row_observations(),
            action_set=_row_actions(),
            transition_rules={"family": "policy_event_simulator", "policyRepresentation": "PolicySpec"},
            goal=_sorting_goal("declared_by_condition"),
            perturbation={"inherits": "E01/E02 selected no-Frozen, Frozen Cell, and chimera cases"},
            scheduler={"family": "E02-compatible seeded event scheduler"},
            measurement_functions=_sorting_measurements(["policy_interface_replay_exactness"]),
            artifact_paths=[
                _artifact(previous, "E03", "research_steps", "S01", "policy_catalog.parquet"),
                _artifact(previous, "E03", "research_steps", "S01", "policy_interface_validation.parquet"),
            ],
            status_paths=[_status(previous, "E03", "S01")],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E03", "research_steps", "S01", "policy_catalog.csv")),
        ),
        _record(
            world_id="E03_S02_S05_policy_dsl_corpus_world",
            experiment_id="E03",
            source_steps=["S02", "S03", "S04", "S05"],
            title="Rule-DSL policy corpus task world",
            world_family="policy_morphospace",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", "toy_and_100_cell"),
            local_observations=_row_observations(["dsl_state_keys", "tie_breaker_rng"]),
            action_set={"primitiveActions": ["wait", "swap_left", "swap_right", "swap_target", "state_update"], "legality": "DSL rule predicates and action guards"},
            transition_rules={"family": "rule_dsl_policy_execution", "policyCount": _row_count(_artifact(previous, "E03", "research_steps", "S05", "policy_corpus.csv"))},
            goal=_sorting_goal("increasing"),
            perturbation={"policyGeneration": ["classic_templates", "seed_import", "mutation", "recombination"]},
            scheduler={"family": "policy_event_or_batch_scheduler", "seeded": True},
            measurement_functions=_sorting_measurements(["competence_vector"]),
            artifact_paths=[
                _artifact(previous, "E03", "research_steps", "S03", "classic_policy_templates.parquet"),
                _artifact(previous, "E03", "research_steps", "S04", "competence_vector_schema.json"),
                _artifact(previous, "E03", "research_steps", "S05", "policy_corpus.parquet"),
            ],
            status_paths=[_status(previous, "E03", step) for step in ["S02", "S03", "S04", "S05"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E03", "research_steps", "S05", "policy_corpus.csv")),
        ),
        _record(
            world_id="E03_S06_S09_batch_sweep_and_quality_diversity_world",
            experiment_id="E03",
            source_steps=["S06", "S07", "S08", "S09"],
            title="Batch policy sweeps and quality-diversity morphospace",
            world_family="policy_morphospace",
            substrate_class="row_1d_batch",
            state_space=_base_state("row_1d", "batched_conditions"),
            local_observations=_row_observations(["dsl_features"]),
            action_set={"primitiveActions": ["wait", "swap_left", "swap_right", "swap_target"], "legality": "batch simulator CPU/GPU validated policy semantics"},
            transition_rules={"family": "batched_policy_sweep", "validation": "CPU reference agreement"},
            goal=_sorting_goal("increasing"),
            perturbation={"sweepAxes": ["policy_features", "phase_boundaries", "quality_diversity_candidates"]},
            scheduler={"family": "batched_seeded_scheduler", "hardware": "CPU/GPU when available"},
            measurement_functions=_sorting_measurements(["quality_diversity_elite", "phase_boundary"]),
            artifact_paths=[
                _artifact(previous, "E03", "research_steps", "S06", "batch_simulation_validation.parquet"),
                _artifact(previous, "E03", "research_steps", "S08", "elite_policy_index.parquet"),
                _artifact(previous, "E03", "research_steps", "S09", "phase_boundary_policy_task_summary.parquet"),
            ],
            status_paths=[_status(previous, "E03", step) for step in ["S06", "S07", "S08", "S09"]],
            replayability="derived",
            metadata_completeness="complete",
            source_record_count=sum(
                value or 0
                for value in [
                    _row_count(_artifact(previous, "E03", "research_steps", "S08", "elite_policy_index.csv")),
                    _row_count(_artifact(previous, "E03", "research_steps", "S09", "phase_boundary_policy_task_summary.csv")),
                ]
            ),
        ),
        _record(
            world_id="E03_S10_S15_behavior_atlas_world",
            experiment_id="E03",
            source_steps=["S10", "S11", "S12", "S13", "S14", "S15"],
            title="Behavior embedding, universality, ablation, and atlas world panel",
            world_family="behavior_atlas",
            substrate_class="abstract_behavior_panel",
            state_space={"substrate": "behavior_matrix", "elements": "policy-by-task competence vectors and selected validation runs"},
            local_observations={"locality": "derived_feature_space", "fields": ["competence_vector", "source_policy_features"], "globalInformationExcluded": False},
            action_set={"primitiveActions": ["embedding", "classification", "ablation", "frontier_selection"], "legality": "analysis operations only"},
            transition_rules={"family": "post_simulation_behavior_analysis"},
            goal={"goalType": "behavioral_taxonomy", "predicate": "policies near each other when behaviorally similar"},
            perturbation={"analysisPanels": ["feature_ablation", "classic_vs_discovered", "frontier_candidates"]},
            scheduler={"family": "not_applicable_analysis_layer"},
            measurement_functions=[{"name": "embedding_coordinates", "type": "analysis"}, {"name": "universality_class", "type": "analysis"}],
            artifact_paths=[
                _artifact(previous, "E03", "research_steps", "S10", "policy_embeddings.parquet"),
                _artifact(previous, "E03", "research_steps", "S11", "policy_class_assignments.parquet"),
                _artifact(previous, "E03", "research_steps", "S15", "policy_atlas_catalog.parquet"),
            ],
            status_paths=[_status(previous, "E03", step) for step in ["S10", "S11", "S12", "S13", "S14", "S15"]],
            replayability="summary_only",
            metadata_completeness="exception_documented",
            source_record_count=_row_count(_artifact(previous, "E03", "research_steps", "S15", "policy_atlas_catalog.csv")),
            exceptions_or_gaps=["This is an analysis/atlas panel over policy behavior, not an independent transition world."],
        ),
    ]


def _e04_records(previous: Path) -> list[dict[str, Any]]:
    return [
        _record(
            world_id="E04_S01_memory_augmented_sorting_world",
            experiment_id="E04",
            source_steps=["S01", "S09"],
            title="Memory-augmented local sorting world",
            world_family="memory_repair",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", "varies", ["scalar_value", "algotype", "memory_state"]),
            local_observations=_row_observations(["bounded_counter", "neighbor_history"]),
            action_set=_row_actions(["memory_update"]),
            transition_rules={"family": "memory_wrapped_policy", "memoryVariants": _unique_values(_artifact(previous, "E04", "research_steps", "S01", "memory_variant_catalog.csv"), "memoryVariant")},
            goal=_sorting_goal("increasing"),
            perturbation={"memoryAblations": "S09"},
            scheduler={"family": "seeded_policy_event_scheduler"},
            measurement_functions=_sorting_measurements(["memory_ablation_effect"]),
            artifact_paths=[
                _artifact(previous, "E04", "research_steps", "S01", "memory_variant_catalog.parquet"),
                _artifact(previous, "E04", "research_steps", "S09", "memory_ablation_validation.parquet"),
            ],
            status_paths=[_status(previous, "E04", step) for step in ["S01", "S09"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E04", "research_steps", "S01", "memory_variant_catalog.csv")),
        ),
        _record(
            world_id="E04_S02_signal_communication_world",
            experiment_id="E04",
            source_steps=["S02", "S10", "S12"],
            title="Local signal communication and tissue-field proxy world",
            world_family="communication_repair",
            substrate_class="row_1d_with_signal_field",
            state_space=_base_state("row_1d", "varies", ["scalar_value", "algotype", "signal_channels"]),
            local_observations=_row_observations(["local_signal_channels", "diffused_neighbor_signal"]),
            action_set=_row_actions(["emit_signal", "diffuse_signal"]),
            transition_rules={"family": "signal_wrapped_policy", "signalVariants": _unique_values(_artifact(previous, "E04", "research_steps", "S02", "signal_variant_catalog.csv"), "signalVariant")},
            goal=_sorting_goal("increasing"),
            perturbation={"communicationAblations": "S10", "fieldPredictorPanel": "S12"},
            scheduler={"family": "sense_act_emit_diffuse"},
            measurement_functions=_sorting_measurements(["signal_ablation_effect", "field_predictor_score"]),
            artifact_paths=[
                _artifact(previous, "E04", "research_steps", "S02", "signal_variant_catalog.parquet"),
                _artifact(previous, "E04", "research_steps", "S10", "communication_ablation_validation.parquet"),
                _artifact(previous, "E04", "research_steps", "S12", "field_predictor_validation.parquet"),
            ],
            status_paths=[_status(previous, "E04", step) for step in ["S02", "S10", "S12"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E04", "research_steps", "S02", "signal_variant_catalog.csv")),
        ),
        _record(
            world_id="E04_S03_repairable_frozen_world",
            experiment_id="E04",
            source_steps=["S03", "S11"],
            title="Repairable Frozen Cell world",
            world_family="repair",
            substrate_class="row_1d_repairable_defect",
            state_space=_base_state("row_1d", "varies", ["scalar_value", "algotype", "repairable_frozen_state"]),
            local_observations=_row_observations(["blocked_history", "repair_signal"]),
            action_set=_row_actions(["repair_nudge", "repair_attempt"]),
            transition_rules={"family": "repairable_frozen_dynamics", "defectState": "can recover or be nudged under local rules"},
            goal=_sorting_goal("increasing", "restore_order_after_defect"),
            perturbation={"defect": "repairable_frozen", "repairConfigs": "S03"},
            scheduler={"family": "seeded_repair_task_scheduler"},
            measurement_functions=_sorting_measurements(["repair_success", "time_to_repair", "competence_proxy"]),
            artifact_paths=[
                _artifact(previous, "E04", "research_steps", "S03", "repair_task_run_summary.parquet"),
                _artifact(previous, "E04", "research_steps", "S11", "competence_proxy_validation.parquet"),
            ],
            status_paths=[_status(previous, "E04", step) for step in ["S03", "S11"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E04", "research_steps", "S03", "repair_task_run_summary.csv")),
        ),
        _record(
            world_id="E04_S04_S05_homeostasis_and_damage_world",
            experiment_id="E04",
            source_steps=["S04", "S05", "S08"],
            title="Fatigue, damage, and homeostasis world",
            world_family="homeostasis",
            substrate_class="row_1d_dynamic",
            state_space=_base_state("row_1d", "variable_length", ["scalar_value", "damage_state", "fatigue_state", "homeostatic_length"]),
            local_observations=_row_observations(["damage_state", "fatigue_state", "local_length_signal"]),
            action_set=_row_actions(["insert", "delete", "recover", "homeostatic_noop"]),
            transition_rules={"family": "dynamic_homeostatic_task", "taskIds": _unique_values(_artifact(previous, "E04", "research_steps", "S05", "homeostatic_task_catalog.csv"), "taskId")},
            goal={"goalType": "homeostasis", "predicate": "maintain sortedness and length/identity constraints after perturbations", "energy": "failure_duration and recovery score"},
            perturbation={"families": ["fatigue_damage", "swap", "insert", "delete"]},
            scheduler={"family": "tick_schedule", "horizonTicks": "task_specific"},
            measurement_functions=_sorting_measurements(["homeostasis_success", "failure_duration"]),
            artifact_paths=[
                _artifact(previous, "E04", "research_steps", "S04", "fatigue_validation_results.parquet"),
                _artifact(previous, "E04", "research_steps", "S05", "homeostatic_task_catalog.parquet"),
                _artifact(previous, "E04", "research_steps", "S08", "homeostasis_tick_records.parquet"),
            ],
            status_paths=[_status(previous, "E04", step) for step in ["S04", "S05", "S08"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E04", "research_steps", "S05", "homeostatic_task_catalog.csv")),
        ),
        _record(
            world_id="E04_S06_S15_learning_evolution_synthesis_world",
            experiment_id="E04",
            source_steps=["S06", "S07", "S08", "S13", "S14", "S15"],
            title="Local learning, evolution, transfer, and synthesis world panel",
            world_family="adaptive_repair",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", "varies", ["scalar_value", "learning_state", "memory_state", "signal_state"]),
            local_observations=_row_observations(["local_reward_terms", "recent_failure", "signal_channels"]),
            action_set=_row_actions(["adapt_action_probability", "emit_signal", "repair_attempt"]),
            transition_rules={"family": "local_learning_and_evolved_repair_policy", "oracleConstraint": "global oracle disallowed where audited"},
            goal={"goalType": "repair_competence", "predicate": "improve repair/homeostasis proxies under local information constraints"},
            perturbation={"trainingAndTransfer": ["learning", "evolution", "overfitting_transfer", "centralized_comparison"]},
            scheduler={"family": "validation_and_search_schedulers", "workers": "source_step_specific"},
            measurement_functions=_sorting_measurements(["repair_competence_proxy", "transfer_score", "centralized_comparison_delta"]),
            artifact_paths=[
                _artifact(previous, "E04", "research_steps", "S06", "learning_variant_catalog.parquet"),
                _artifact(previous, "E04", "results", "e04_evolved_repair_policies.parquet"),
                _artifact(previous, "E04", "research_steps", "S15", "handoff_policy_catalog.parquet"),
                _artifact(previous, "E04", "research_steps", "S15", "evidence_source_catalog.parquet"),
            ],
            status_paths=[_status(previous, "E04", step) for step in ["S06", "S07", "S08", "S13", "S14", "S15"]],
            replayability="partial",
            metadata_completeness="partial",
            source_record_count=_row_count(_artifact(previous, "E04", "research_steps", "S15", "handoff_policy_catalog.csv")),
            exceptions_or_gaps=["Search/evolution history is summarized in handoff tables; exact search replay may require source-step configs and seeds."],
        ),
    ]


def _e05_records(previous: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    substrate_path = _artifact(previous, "E05", "research_steps", "S01", "substrate_specs.csv")
    for row in _read_rows(substrate_path):
        label = str(row.get("substrate_label") or row.get("substrateType"))
        records.append(
            _record(
                world_id=f"E05_S01_substrate_{label}",
                experiment_id="E05",
                source_steps=["S01"],
                title=f"E05 substrate abstraction: {label}",
                world_family="morphology_substrate",
                substrate_class=label,
                state_space={"substrate": row.get("substrateType"), "nodeCount": row.get("nodeCount"), "edgeCount": row.get("edgeCount"), "boundary": row.get("boundary"), "metadata": row.get("metadata")},
                local_observations={"locality": "graph_neighbors", "fields": ["node_identity", "neighbor_identities", "occupancy"], "globalInformationExcluded": True},
                action_set={"primitiveActions": ["wait", "swap_with_neighbor", "move_to_adjacent_empty_when_allowed"], "legality": "substrate adjacency and occupancy constraints"},
                transition_rules={"family": "substrate_generalization", "conservation": "occupancy and identity conservation in S01 toy simulations"},
                goal={"goalType": "substrate_validity", "predicate": "legal local moves conserve identities and valid adjacency"},
                perturbation={"none": True},
                scheduler={"family": "toy_local_update_scheduler"},
                measurement_functions=[{"name": "adjacency_symmetry", "type": "validation"}, {"name": "identity_conservation", "type": "validation"}],
                artifact_paths=[_artifact(previous, "E05", "research_steps", "S01", "substrate_specs.parquet"), _artifact(previous, "E05", "research_steps", "S01", "substrate_validation_results.parquet")],
                status_paths=[_status(previous, "E05", "S01")],
                replayability="direct",
                metadata_completeness="complete",
                source_record_count=1,
            )
        )

    target_path = _artifact(previous, "E05", "research_steps", "S03", "target_catalog.csv")
    for row in _read_rows(target_path):
        target_id = str(row.get("target_id"))
        records.append(
            _record(
                world_id=f"E05_S03_target_{target_id}",
                experiment_id="E05",
                source_steps=["S02", "S03", "S04", "S05"],
                title=f"E05 target morphology: {row.get('title')}",
                world_family="morphology_target",
                substrate_class=str(row.get("substrate_type")),
                state_space={"substrate": row.get("substrate_type"), "nodeCount": row.get("node_count"), "edgeCount": row.get("edge_count"), "identitySchema": "e05_s02_identity_schema.v1"},
                local_observations={"locality": "graph_neighbors", "fields": ["identity_components", "neighbor_preferences", "local_target_free_features"], "globalInformationExcluded": True},
                action_set={"primitiveActions": ["wait", "swap_neighbor", "nonconservative_actions_when_later_steps_allow"], "legality": "S04 action semantics and target constraints"},
                transition_rules={"family": "target_morphology_local_actions", "motif": row.get("motif")},
                goal={"goalType": "target_morphology_energy", "targetId": target_id, "energy": "weighted identity and neighborhood target error"},
                perturbation={"initialization": "scrambled or benchmark specific"},
                scheduler={"family": "morphology_local_scheduler"},
                measurement_functions=[
                    {"name": "target_energy", "type": "computational_proxy"},
                    {"name": "boundary_error", "type": "computational_proxy"},
                    {"name": "topology_error", "type": "computational_proxy"},
                ],
                artifact_paths=[
                    _artifact(previous, "E05", "research_steps", "S03", "target_catalog.parquet"),
                    _artifact(previous, "E05", "research_steps", "S05", "metric_catalog.parquet"),
                    _artifact(previous, "E05", "results", "e05_metric_validation.parquet"),
                ],
                status_paths=[_status(previous, "E05", step) for step in ["S02", "S03", "S04", "S05"]],
                replayability="direct",
                metadata_completeness="complete",
                source_record_count=1,
            )
        )

    perturbation_path = _artifact(previous, "E05", "research_steps", "S08", "perturbation_catalog.csv")
    perturb_rows = _read_rows(perturbation_path)
    perturb_counts = Counter(str(row.get("perturbation_family")) for row in perturb_rows)
    for family, count in sorted(perturb_counts.items()):
        records.append(
            _record(
                world_id=f"E05_S08_perturbation_{family}",
                experiment_id="E05",
                source_steps=["S07", "S08"],
                title=f"E05 regeneration perturbation family: {family}",
                world_family="morphology_repair",
                substrate_class="grid_or_graph_morphology",
                state_space={"substrate": "target-dependent", "occupancy": "may be conservative or nonconservative", "cellCountDelta": "family_specific"},
                local_observations={"locality": "graph_neighbors", "fields": ["observed_identity", "missing_or_extra_positions", "local_neighbors"], "globalInformationExcluded": True},
                action_set={"primitiveActions": ["swap_neighbor", "insert", "delete", "fill_missing", "remove_extra"], "legality": "benchmark-specific repair actions"},
                transition_rules={"family": "regeneration_or_scrambled_recovery", "perturbationFamily": family},
                goal={"goalType": "repair_to_target_morphology", "energy": "target error reduction"},
                perturbation={"perturbationFamily": family, "rowCount": count},
                scheduler={"family": "morphology_repair_scheduler"},
                measurement_functions=[{"name": "repair_success", "type": "computational_proxy"}, {"name": "relative_error_reduction", "type": "computational_proxy"}],
                artifact_paths=[_artifact(previous, "E05", "research_steps", "S08", "perturbation_catalog.parquet"), _artifact(previous, "E05", "results", "e05_regeneration_tests.parquet")],
                status_paths=[_status(previous, "E05", step) for step in ["S07", "S08"]],
                replayability="direct",
                metadata_completeness="complete",
                source_record_count=count,
            )
        )

    symmetry_path = _artifact(previous, "E05", "research_steps", "S10", "symmetry_task_catalog.csv")
    for row in _read_rows(symmetry_path):
        task_id = str(row.get("task_id"))
        records.append(
            _record(
                world_id=f"E05_S10_symmetry_{task_id}",
                experiment_id="E05",
                source_steps=["S10"],
                title=f"E05 symmetry-breaking task: {task_id}",
                world_family="morphology_symmetry",
                substrate_class="square_grid_2d",
                state_space={"substrate": "square_grid_2d", "gridSize": row.get("grid_size"), "initialClass": "uniform_neutral_square"},
                local_observations={"locality": "grid_neighbors", "fields": ["neutral_identity", "neighbor_identity"], "globalInformationExcluded": True},
                action_set={"primitiveActions": ["wait", "swap_neighbor", "differentiate_local_identity_when_policy_allows"], "legality": "task-specific local action rules"},
                transition_rules={"family": "symmetry_breaking", "motif": row.get("motif")},
                goal={"goalType": "symmetry_breaking_target", "targetKind": row.get("target_kind"), "targetAxis": row.get("target_axis")},
                perturbation={"symmetricInitialClass": "uniform_neutral_square"},
                scheduler={"family": "morphology_local_scheduler"},
                measurement_functions=[{"name": "symmetry_breaking_success", "type": "computational_proxy"}],
                artifact_paths=[_artifact(previous, "E05", "research_steps", "S10", "symmetry_task_catalog.parquet"), _artifact(previous, "E05", "results", "e05_symmetry_breaking.parquet")],
                status_paths=[_status(previous, "E05", "S10")],
                replayability="direct",
                metadata_completeness="complete",
                source_record_count=1,
            )
        )

    bench_path = _artifact(previous, "E05", "research_steps", "S15", "benchmark_suite_config_catalog.csv")
    for row in _read_rows(bench_path):
        benchmark_id = str(row.get("benchmark_id"))
        records.append(
            _record(
                world_id=f"E05_S15_benchmark_{benchmark_id}",
                experiment_id="E05",
                source_steps=["S15"],
                title=f"E05 benchmark suite config: {row.get('title')}",
                world_family="morphology_benchmark",
                substrate_class="benchmark_panel",
                state_space={"substrate": "benchmark-selected", "benchmarkFamily": row.get("benchmark_family")},
                local_observations={"locality": "inherited_from_selected_worlds", "fields": ["benchmark_selected_features"], "globalInformationExcluded": "varies_by_policy_control_class"},
                action_set={"primitiveActions": ["inherited_from_selected_worlds"], "legality": "benchmark config selectors"},
                transition_rules={"family": "benchmark_suite_membership", "benchmarkId": benchmark_id},
                goal={"goalType": row.get("primary_goal"), "predicate": "benchmark-specific success or error reduction"},
                perturbation={"benchmarkSelectors": row.get("selector_count"), "caveatsJson": row.get("caveats_json")},
                scheduler={"family": "inherited_from_selected_source_runs"},
                measurement_functions=[{"name": name, "type": "computational_proxy"} for name in _safe_json_list(row.get("metrics_json"))],
                artifact_paths=[
                    _artifact(previous, "E05", "research_steps", "S15", "benchmark_suite_config_catalog.parquet"),
                    _artifact(previous, "E05", "research_steps", "S15", "morphology_benchmark_suite_rows.parquet"),
                    _artifact(previous, "E05", "results", "e05_morphology_benchmarks.parquet"),
                ],
                status_paths=[_status(previous, "E05", "S15")],
                replayability="summary_only",
                metadata_completeness="exception_documented",
                source_record_count=int(row.get("expected_min_rows") or 0),
                exceptions_or_gaps=["Benchmark configs select rows from source worlds; transition semantics must be inherited from those source records."],
            )
        )
    return records


def _safe_json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(parsed, list):
        out: list[str] = []
        for item in parsed:
            if isinstance(item, list):
                out.extend(str(sub) for sub in item)
            else:
                out.append(str(item))
        return out
    return [str(parsed)]


def _e06_records(previous: Path) -> list[dict[str, Any]]:
    return [
        _record(
            world_id="E06_S01_pure_policy_panel_world",
            experiment_id="E06",
            source_steps=["S01"],
            title="Pure-policy chimerism panel anchor",
            world_family="chimera",
            substrate_class="row_1d",
            state_space=_base_state("row_1d", 100, ["scalar_value", "candidate_policy_id"]),
            local_observations=_row_observations(["policy_metadata"]),
            action_set=_row_actions(),
            transition_rules={"family": "pure_policy_validation", "policyPanelRows": _row_count(_artifact(previous, "E06", "research_steps", "S01", "algotype_panel.csv"))},
            goal=_sorting_goal("increasing"),
            perturbation={"none": "pure controls before mixing"},
            scheduler={"family": "seeded_policy_event_scheduler"},
            measurement_functions=_sorting_measurements(["pure_policy_validation"]),
            artifact_paths=[_artifact(previous, "E06", "research_steps", "S01", "algotype_panel.parquet"), _artifact(previous, "E06", "research_steps", "S01", "pure_policy_validation.parquet")],
            status_paths=[_status(previous, "E06", "S01")],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E06", "research_steps", "S01", "algotype_panel.csv")),
        ),
        _record(
            world_id="E06_S02_S03_mixture_arrangement_worlds",
            experiment_id="E06",
            source_steps=["S02", "S03"],
            title="Mixture-ratio and initial-arrangement chimeric worlds",
            world_family="chimera",
            substrate_class="mixed_1d_chimera",
            state_space=_base_state("row_1d", 100, ["scalar_value", "policy_id", "initial_arrangement_class"]),
            local_observations=_row_observations(["neighbor_policy_id", "interface_state"]),
            action_set=_row_actions(),
            transition_rules={"family": "mixed_policy_event_simulation", "arrangement": "random, block, gradient, and selected sensitive layouts"},
            goal=_sorting_goal("increasing", "shared_or_later_goal_mode"),
            perturbation={"mixtureRatios": "S02", "initialArrangements": "S03"},
            scheduler={"family": "seeded_policy_event_scheduler"},
            measurement_functions=_sorting_measurements(["interface_density", "policy_position_change"]),
            artifact_paths=[_artifact(previous, "E06", "results", "e06_mixture_ratios.parquet"), _artifact(previous, "E06", "results", "e06_initial_arrangements.parquet")],
            status_paths=[_status(previous, "E06", step) for step in ["S02", "S03"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=sum(value or 0 for value in [_row_count(_artifact(previous, "E06", "results", "e06_mixture_ratios.csv")), _row_count(_artifact(previous, "E06", "results", "e06_initial_arrangements.csv"))]),
        ),
        _record(
            world_id="E06_S04_S05_goal_compatibility_worlds",
            experiment_id="E06",
            source_steps=["S04", "S05"],
            title="Chimeric goal-compatibility worlds",
            world_family="chimera_goal_compatibility",
            substrate_class="mixed_1d_chimera",
            state_space=_base_state("row_1d", 100, ["scalar_value", "policy_id", "per_policy_goal_spec"]),
            local_observations=_row_observations(["own_goal_direction", "neighbor_policy_id"]),
            action_set=_row_actions(),
            transition_rules={"family": "goal_mode_chimera", "goalModes": _unique_values(_artifact(previous, "E06", "research_steps", "S05", "compatibility_mode_summary.csv"), "goalMode")},
            goal={"goalType": "compatible_partial_or_opposed_policy_goals", "predicate": "target-quality vector by policy goal mode"},
            perturbation={"goalMode": ["same_goal", "partially_compatible", "opposite_goal", "other source modes"]},
            scheduler={"family": "seeded_policy_event_scheduler"},
            measurement_functions=_sorting_measurements(["compatibility_composite_score", "minority_score", "compromise_score"]),
            artifact_paths=[_artifact(previous, "E06", "results", "e06_goal_compatibility.parquet"), _artifact(previous, "E06", "results", "e06_compatibility_metrics.parquet")],
            status_paths=[_status(previous, "E06", step) for step in ["S04", "S05"]],
            replayability="direct",
            metadata_completeness="complete",
            source_record_count=_row_count(_artifact(previous, "E06", "results", "e06_compatibility_metrics.csv")),
        ),
        _record(
            world_id="E06_S06_S08_dominance_mosaic_interface_worlds",
            experiment_id="E06",
            source_steps=["S06", "S07", "S08"],
            title="Dominance, mosaic, and interface-rule chimeric worlds",
            world_family="chimera_governance",
            substrate_class="mixed_1d_chimera",
            state_space=_base_state("row_1d", 100, ["scalar_value", "policy_id", "interface_rule_state"]),
            local_observations=_row_observations(["neighbor_policy_id", "interface_rule_context"]),
            action_set=_row_actions(["interface_rule_action"]),
            transition_rules={"family": "dominance_mosaic_interface_variants"},
            goal={"goalType": "dominance_or_mosaic_proxy", "predicate": "winner, interface stability, or mosaic class proxies"},
            perturbation={"variantFamilies": ["dominance_contest", "mosaic_formation", "interface_rule"]},
            scheduler={"family": "seeded_policy_event_scheduler"},
            measurement_functions=_sorting_measurements(["dominance_proxy", "mosaic_class", "interface_stability"]),
            artifact_paths=[
                _artifact(previous, "E06", "results", "e06_dominance_hierarchy.parquet"),
                _artifact(previous, "E06", "results", "e06_mosaic_classifications.parquet"),
                _artifact(previous, "E06", "results", "e06_interface_rules.parquet"),
            ],
            status_paths=[_status(previous, "E06", step) for step in ["S06", "S07", "S08"]],
            replayability="derived",
            metadata_completeness="complete",
            source_record_count=sum(value or 0 for value in [_row_count(_artifact(previous, "E06", "results", "e06_dominance_hierarchy.csv")), _row_count(_artifact(previous, "E06", "results", "e06_mosaic_classifications.csv")), _row_count(_artifact(previous, "E06", "results", "e06_interface_rules.csv"))]),
        ),
        _record(
            world_id="E06_S09_S14_governance_graft_mutant_history_intervention_worlds",
            experiment_id="E06",
            source_steps=["S09", "S10", "S11", "S12", "S13", "S14"],
            title="Governance, graft, mutant, history, causal, and intervention chimeric worlds",
            world_family="chimera_intervention",
            substrate_class="mixed_1d_chimera",
            state_space=_base_state("row_1d", 100, ["scalar_value", "policy_id", "governance_state", "graft_or_mutant_label", "history_protocol"]),
            local_observations=_row_observations(["local_vote_signal", "graft_boundary", "mutant_clone_marker", "history_phase"]),
            action_set=_row_actions(["governance_vote", "broad_organizer_proxy", "graft_insert", "mutant_clone_insert", "intervention_action"]),
            transition_rules={"family": "chimeric_control_and_intervention_variants"},
            goal={"goalType": "rescue_or_control_proxy", "predicate": "improve target quality or compatibility under declared caveats"},
            perturbation={"variantFamilies": ["governance", "graft", "mutant_clone", "developmental_history", "intervention"]},
            scheduler={"family": "seeded_policy_event_scheduler_with_history_protocols"},
            measurement_functions=_sorting_measurements(["rescue_success_proxy", "cost_proxy", "side_effect_penalty", "causal_prediction_score"]),
            artifact_paths=[
                _artifact(previous, "E06", "results", "e06_governance_mechanisms.parquet"),
                _artifact(previous, "E06", "results", "e06_graft_experiments.parquet"),
                _artifact(previous, "E06", "results", "e06_mutant_clone_experiments.parquet"),
                _artifact(previous, "E06", "results", "e06_developmental_history.parquet"),
                _artifact(previous, "E06", "results", "e06_intervention_search.parquet"),
                _artifact(previous, "E06", "results", "e06_causal_predictive_models.parquet"),
            ],
            status_paths=[_status(previous, "E06", step) for step in ["S09", "S10", "S11", "S12", "S13", "S14"]],
            replayability="partial",
            metadata_completeness="partial",
            source_record_count=sum(value or 0 for value in [_row_count(_artifact(previous, "E06", "results", name)) for name in ["e06_governance_mechanisms.csv", "e06_graft_experiments.csv", "e06_mutant_clone_experiments.csv", "e06_developmental_history.csv", "e06_intervention_search.csv", "e06_causal_predictive_models.csv"]]),
            exceptions_or_gaps=["Some intervention and causal rows are post hoc or held-out validation summaries; replay requires source-step condition generators and explicit Chief approval before running S02+."],
        ),
        _record(
            world_id="E06_S15_chimeric_playbook_world_panel",
            experiment_id="E06",
            source_steps=["S15"],
            title="Chimeric-control playbook world panel",
            world_family="chimera_atlas",
            substrate_class="abstract_behavior_panel",
            state_space={"substrate": "phase_diagram_and_recommendation_panel", "elements": "world summaries, evidence links, caveat register"},
            local_observations={"locality": "analysis_layer", "fields": ["phase_axis", "recommendation", "evidence_traceability"], "globalInformationExcluded": False},
            action_set={"primitiveActions": ["summarize", "recommend", "link_evidence"], "legality": "analysis operations only"},
            transition_rules={"family": "post_simulation_playbook_synthesis"},
            goal={"goalType": "bounded_chimeric_control_synthesis", "predicate": "recommendations retain caveats and evidence links"},
            perturbation={"none": "analysis layer"},
            scheduler={"family": "not_applicable_analysis_layer"},
            measurement_functions=[{"name": "recommendation_traceability", "type": "validation"}, {"name": "caveat_preservation", "type": "validation"}],
            artifact_paths=[_artifact(previous, "E06", "results", "e06_phase_diagram_summary.parquet"), _artifact(previous, "E06", "results", "e06_chimeric_control_recommendations.parquet"), _artifact(previous, "E06", "results", "e06_evidence_traceability.parquet")],
            status_paths=[_status(previous, "E06", "S15")],
            replayability="summary_only",
            metadata_completeness="exception_documented",
            source_record_count=_row_count(_artifact(previous, "E06", "results", "e06_chimeric_control_recommendations.csv")),
            exceptions_or_gaps=["This is a synthesis and playbook layer; transition semantics must be inherited from linked E06 S01-S14 worlds."],
        ),
    ]


def build_world_catalog(previous_artifacts_dir: str | Path = "/previous-artifacts") -> list[dict[str, Any]]:
    previous = Path(previous_artifacts_dir)
    records: list[dict[str, Any]] = []
    records.extend(_e01_records(previous))
    records.extend(_e02_records(previous))
    records.extend(_e03_records(previous))
    records.extend(_e04_records(previous))
    records.extend(_e05_records(previous))
    records.extend(_e06_records(previous))
    return records


def catalog_to_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for column in JSON_COLUMNS:
            row[column] = compact_json(row.get(column, [] if column.endswith("Paths") or column == "sourceStepIds" else {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered = list(REQUIRED_RECORD_FIELDS) + [
        "metadataCompleteness",
        "sourceRecordCount",
        "exceptionsOrGaps",
        "migrationHints",
        "notes",
    ]
    seen: set[str] = set()
    columns = []
    for column in ordered + list(df.columns):
        if column in df.columns and column not in seen:
            seen.add(column)
            columns.append(column)
    return df[columns].sort_values(["experimentId", "worldId"]).reset_index(drop=True)


def _status_success(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid_json:{exc}"
    success = data.get("success")
    status = str(data.get("status", "")).lower()
    if success is True or status == "completed":
        return True, data.get("validationResult", "completed")
    return False, f"not_success:{success}:{status}"


def validate_world_catalog(records: Sequence[Mapping[str, Any]], schema: Mapping[str, Any] | None = None) -> pd.DataFrame:
    del schema
    rows: list[dict[str, Any]] = []
    seen_world_ids: set[str] = set()
    for record in records:
        world_id = str(record.get("worldId", "UNKNOWN"))
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record or record.get(field) in (None, "", [])]
        rows.append(
            {
                "worldId": world_id,
                "checkId": "required_fields",
                "success": not missing,
                "severity": "error" if missing else "info",
                "detail": "missing=" + ",".join(missing) if missing else "all required fields populated",
            }
        )

        duplicate = world_id in seen_world_ids
        seen_world_ids.add(world_id)
        rows.append(
            {
                "worldId": world_id,
                "checkId": "unique_world_id",
                "success": not duplicate,
                "severity": "error" if duplicate else "info",
                "detail": "duplicate worldId" if duplicate else "worldId is unique so far",
            }
        )

        tuple_errors: list[str] = []
        for field in TUPLE_FIELDS:
            value = record.get(field)
            if field == "measurementFunctions":
                if not isinstance(value, list) or not value:
                    tuple_errors.append(field)
            elif not isinstance(value, Mapping) or not value:
                tuple_errors.append(field)
        rows.append(
            {
                "worldId": world_id,
                "checkId": "formal_tuple_shape",
                "success": not tuple_errors,
                "severity": "error" if tuple_errors else "info",
                "detail": "bad tuple fields=" + ",".join(tuple_errors) if tuple_errors else "tuple fields have expected object/list shapes",
            }
        )

        artifact_paths = [Path(path) for path in record.get("upstreamArtifactPaths", [])]
        missing_artifacts = [str(path) for path in artifact_paths if not path.exists()]
        rows.append(
            {
                "worldId": world_id,
                "checkId": "upstream_artifacts_exist",
                "success": bool(artifact_paths) and not missing_artifacts,
                "severity": "error" if missing_artifacts or not artifact_paths else "info",
                "detail": "missing=" + compact_json(missing_artifacts) if missing_artifacts else f"{len(artifact_paths)} artifact links resolved",
            }
        )

        status_paths = [Path(path) for path in record.get("upstreamStatusPaths", [])]
        status_results = [_status_success(path) for path in status_paths]
        failed_statuses = [str(path) for path, (ok, _detail) in zip(status_paths, status_results, strict=False) if not ok]
        rows.append(
            {
                "worldId": world_id,
                "checkId": "upstream_status_success",
                "success": bool(status_paths) and not failed_statuses,
                "severity": "error" if failed_statuses or not status_paths else "info",
                "detail": "failed=" + compact_json(failed_statuses) if failed_statuses else f"{len(status_paths)} upstream statuses completed",
            }
        )

        partial = record.get("replayability") in {"partial", "summary_only"} or record.get("metadataCompleteness") != "complete"
        exceptions = record.get("exceptionsOrGaps", [])
        rows.append(
            {
                "worldId": world_id,
                "checkId": "partial_records_document_exceptions",
                "success": (not partial) or bool(exceptions),
                "severity": "warning" if partial else "info",
                "detail": f"{len(exceptions)} exceptions documented" if partial else "complete record",
            }
        )
    return pd.DataFrame(rows).sort_values(["worldId", "checkId"]).reset_index(drop=True)


def validation_summary(validation: pd.DataFrame, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]
    warnings = validation[(validation["severity"].eq("warning")) & (~validation["success"])]
    return {
        "allHardChecksPassed": hard_failures.empty,
        "hardFailureCount": int(len(hard_failures)),
        "warningFailureCount": int(len(warnings)),
        "validationCheckCount": int(len(validation)),
        "worldRecordCount": int(len(records)),
        "experimentCounts": dict(Counter(str(record["experimentId"]) for record in records)),
        "replayabilityCounts": dict(Counter(str(record["replayability"]) for record in records)),
        "metadataCompletenessCounts": dict(Counter(str(record["metadataCompleteness"]) for record in records)),
    }
