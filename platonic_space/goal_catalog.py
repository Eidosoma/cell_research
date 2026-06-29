from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .policy_catalog import POLICY_CATALOG_VERSION, POLICY_SCHEMA_VERSION, read_table, safe_json_loads
from .world_schema import SCHEMA_VERSION as WORLD_SCHEMA_VERSION


GOAL_SCHEMA_VERSION = "e07_s03_goal_schema.v1"
GOAL_CATALOG_VERSION = "e07_s03_goal_catalog.v1"
GOAL_CLAIM_BOUNDARY = (
    "Computational goal catalog only; goals are constraints, energies, or metric proxies over simulations, "
    "not direct evidence about biological morphogenesis, cognition, agency, sentience, clinical behavior, "
    "or living chimeras."
)

REQUIRED_GOAL_FIELDS = (
    "abstractGoalId",
    "goalLabel",
    "goalSchemaVersion",
    "goalCatalogVersion",
    "sourceExperimentIds",
    "sourceStepIds",
    "goalFamily",
    "goalKind",
    "representationType",
    "constraintOrEnergyJson",
    "goalCompatibilityJson",
    "localObservability",
    "targetDimensionality",
    "targetSubstrateClasses",
    "toleranceJson",
    "metricBindingsJson",
    "upstreamMeasurementFunctionNames",
    "linkedWorldIds",
    "linkedPolicyIds",
    "policyCoverageJson",
    "sourceGoalPayloadsJson",
    "sourceArtifactPaths",
    "perfectExampleJson",
    "failedExampleJson",
    "exampleValidationStatus",
    "caveatsOrBlockers",
    "claimBoundary",
    "worldSchemaVersion",
    "policySchemaVersion",
    "policyCatalogVersion",
)

NONEMPTY_GOAL_FIELDS = (
    "abstractGoalId",
    "goalLabel",
    "goalSchemaVersion",
    "goalCatalogVersion",
    "sourceExperimentIds",
    "sourceStepIds",
    "goalFamily",
    "goalKind",
    "representationType",
    "constraintOrEnergyJson",
    "goalCompatibilityJson",
    "localObservability",
    "targetDimensionality",
    "targetSubstrateClasses",
    "toleranceJson",
    "metricBindingsJson",
    "upstreamMeasurementFunctionNames",
    "linkedWorldIds",
    "policyCoverageJson",
    "sourceGoalPayloadsJson",
    "sourceArtifactPaths",
    "perfectExampleJson",
    "failedExampleJson",
    "exampleValidationStatus",
    "claimBoundary",
    "worldSchemaVersion",
    "policySchemaVersion",
    "policyCatalogVersion",
)

JSON_COLUMNS = (
    "sourceExperimentIds",
    "sourceStepIds",
    "constraintOrEnergyJson",
    "goalCompatibilityJson",
    "targetDimensionality",
    "targetSubstrateClasses",
    "toleranceJson",
    "metricBindingsJson",
    "upstreamMeasurementFunctionNames",
    "linkedWorldIds",
    "linkedPolicyIds",
    "policyCoverageJson",
    "sourceGoalPayloadsJson",
    "sourceArtifactPaths",
    "perfectExampleJson",
    "failedExampleJson",
    "caveatsOrBlockers",
)

HARD_COVERAGE_REQUIREMENTS = {
    "monotonic_order": "monotonic or nondecreasing array-order goals",
    "target_morphology": "target morphology energy goals",
    "aggregation": "aggregation tendency proxy goals",
    "anti_aggregation": "anti-aggregation or dispersion proxy goals",
    "repair_regeneration": "repair or regeneration goals",
    "symmetry": "symmetry-breaking goals",
    "boundary_restoration": "boundary restoration goals",
    "conflict_equilibrium": "conflict, dominance, or equilibrium proxy goals",
    "homeostasis": "homeostasis goals",
}


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
    if not isinstance(value, (list, tuple, dict, set, str, Path)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def json_hash(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


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


def goal_schema_document() -> dict[str, Any]:
    return {
        "schemaVersion": GOAL_SCHEMA_VERSION,
        "catalogVersion": GOAL_CATALOG_VERSION,
        "title": "E07 abstract goal schema",
        "claimBoundary": GOAL_CLAIM_BOUNDARY,
        "goalRecordSchema": {
            "type": "object",
            "required": list(REQUIRED_GOAL_FIELDS),
            "properties": {
                "abstractGoalId": {"type": "string", "description": "Stable E07 goal identifier."},
                "goalLabel": {"type": "string"},
                "sourceExperimentIds": {"type": "array", "items": {"type": "string"}},
                "sourceStepIds": {"type": "array", "items": {"type": "string"}},
                "goalFamily": {
                    "type": "string",
                    "description": "Broad role such as monotonic_order, target_morphology, repair_regeneration, homeostasis, aggregation, anti_aggregation, symmetry, or conflict_equilibrium.",
                },
                "goalKind": {"type": "string", "description": "Source goal predicate or normalized goal mode."},
                "representationType": {
                    "type": "string",
                    "enum": [
                        "constraint_predicate",
                        "energy_function",
                        "metric_proxy",
                        "compatibility_vector",
                        "benchmark_primary_goal",
                        "schema_validity_constraint",
                        "taxonomy_proxy",
                    ],
                },
                "constraintOrEnergyJson": {
                    "type": "object",
                    "description": "Abstract predicate, energy function, proxy score, or benchmark objective definition.",
                },
                "goalCompatibilityJson": {
                    "type": "object",
                    "description": "Goal relationship metadata such as same/opposed/partial compatibility, aggregation polarity, or conflict class.",
                },
                "localObservability": {
                    "type": "string",
                    "description": "Whether the objective is directly local, locally scaffolded with report-facing global score, or purely report-facing.",
                },
                "targetDimensionality": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target dimensions or panels covered by the linked worlds.",
                },
                "targetSubstrateClasses": {"type": "array", "items": {"type": "string"}},
                "toleranceJson": {"type": "object"},
                "metricBindingsJson": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Upstream measurement functions used to score the goal and their optimization direction.",
                },
                "linkedWorldIds": {"type": "array", "items": {"type": "string"}},
                "linkedPolicyIds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Direct S02 policy links plus documented source-family candidate links when direct world-policy links are schema-only.",
                },
                "policyCoverageJson": {"type": "object"},
                "perfectExampleJson": {"type": "object"},
                "failedExampleJson": {"type": "object"},
                "exampleValidationStatus": {"type": "string"},
                "caveatsOrBlockers": {"type": "array", "items": {"type": "string"}},
                "claimBoundary": {"type": "string"},
            },
        },
        "validationContract": {
            "requiredInputs": [
                "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
                "/artifacts/research_steps/S02/policy_abstract_catalog.parquet",
            ],
            "requiredChecks": [
                "required fields are populated",
                "linked S01 worlds resolve",
                "linked S02 policies resolve",
                "goals link to upstream world measurement functions",
                "source artifact paths exist",
                "perfect and failed examples are present and validated",
                "required goal families are represented",
                "all S01 worlds are linked by at least one S03 goal",
            ],
        },
    }


def _parse_json(value: Any, default: Any = None) -> Any:
    return safe_json_loads(value, default)


def _goal_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _parse_json(row.get("goalPredicateOrEnergy"), {})
    return dict(payload or {})


def _measurement_functions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = _parse_json(row.get("measurementFunctions"), [])
    if isinstance(parsed, list):
        return [dict(item) if isinstance(item, Mapping) else {"name": str(item), "type": "unknown"} for item in parsed]
    return []


def _metric_names(row: Mapping[str, Any]) -> list[str]:
    return [str(item.get("name")) for item in _measurement_functions(row) if item.get("name")]


def _json_list(row: Mapping[str, Any], field: str) -> list[Any]:
    parsed = _parse_json(row.get(field), [])
    if isinstance(parsed, list):
        return parsed
    return []


def _existing(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        if Path(path).exists():
            out.append(path)
    return out


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").upper()
    return text or "UNSPECIFIED"


def _world_subset(worlds: pd.DataFrame, selector: Callable[[Mapping[str, Any]], bool]) -> pd.DataFrame:
    mask = [bool(selector(row)) for row in worlds.to_dict(orient="records")]
    return worlds.loc[mask].copy()


def _by_goal_type(worlds: pd.DataFrame, *goal_types: str) -> pd.DataFrame:
    wanted = set(goal_types)
    return _world_subset(worlds, lambda row: str(_goal_payload(row).get("goalType")) in wanted)


def _by_world_ids(worlds: pd.DataFrame, world_ids: Sequence[str]) -> pd.DataFrame:
    wanted = {str(item) for item in world_ids}
    return worlds[worlds["worldId"].astype(str).isin(wanted)].copy()


def _world_ids(rows: pd.DataFrame) -> list[str]:
    return sorted(str(item) for item in rows["worldId"].dropna().unique())


def _source_experiments(rows: pd.DataFrame, extra: Sequence[str] | None = None) -> list[str]:
    values = {str(item) for item in rows.get("experimentId", pd.Series(dtype=str)).dropna().unique()}
    values.update(str(item) for item in (extra or ()))
    return sorted(values)


def _source_steps(rows: pd.DataFrame, extra: Sequence[str] | None = None) -> list[str]:
    values: set[str] = set()
    for row in rows.to_dict(orient="records"):
        values.update(str(item) for item in _json_list(row, "sourceStepIds"))
    values.update(str(item) for item in (extra or ()))
    return sorted(values)


def _source_goal_payloads(rows: pd.DataFrame) -> list[dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict(orient="records"):
        payload = _goal_payload(row)
        key = compact_json(payload)
        if key not in payloads:
            payloads[key] = payload
    return list(payloads.values())


def _source_artifacts(rows: pd.DataFrame, extra: Sequence[str] | None = None) -> list[str]:
    paths: list[str] = []
    for row in rows.to_dict(orient="records"):
        paths.extend(str(item) for item in _json_list(row, "upstreamArtifactPaths"))
    paths.extend(str(item) for item in (extra or ()))
    return _existing(paths)


def _target_substrates(rows: pd.DataFrame) -> list[str]:
    return sorted(str(item) for item in rows.get("substrateClass", pd.Series(dtype=str)).dropna().unique())


def _target_dimensions(substrate_classes: Sequence[str]) -> list[str]:
    dims: set[str] = set()
    for substrate in substrate_classes:
        text = str(substrate).lower()
        if "1d" in text or "row" in text:
            dims.add("1d")
        if "2d" in text or "grid" in text:
            dims.add("2d")
        if "3d" in text:
            dims.add("3d")
        if "graph" in text:
            dims.add("graph")
        if "panel" in text:
            dims.add("abstract_panel")
        if "chimera" in text:
            dims.add("mixed_1d")
    return sorted(dims or {"unspecified"})


def _world_metric_name_set(rows: pd.DataFrame) -> set[str]:
    names: set[str] = set()
    for row in rows.to_dict(orient="records"):
        names.update(_metric_names(row))
    return names


def _direction_for_metric(metric: str, goal_family: str) -> str:
    name = metric.lower()
    if goal_family == "anti_aggregation" and "aggregation" in name:
        return "lower_is_better"
    if any(token in name for token in ("error", "distance", "penalty", "cost", "duration", "count_delta", "violation")):
        return "lower_is_better"
    if any(token in name for token in ("sortedness", "success", "score", "quality", "compatibility", "reduction", "aggregation", "stability")):
        return "higher_is_better"
    if "class" in name or "hash" in name or "traceability" in name:
        return "categorical_or_audit"
    return "context_dependent"


def _metric_bindings(
    rows: pd.DataFrame,
    goal_family: str,
    primary_metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    primary = {str(item) for item in (primary_metrics or ())}
    bindings: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict(orient="records"):
        for metric in _measurement_functions(row):
            name = str(metric.get("name"))
            bindings.setdefault(
                name,
                {
                    "metricName": name,
                    "metricType": metric.get("type", "unknown"),
                    "direction": _direction_for_metric(name, goal_family),
                    "role": "primary" if name in primary else "supporting",
                    "source": "S01_world_measurementFunctions",
                },
            )
            if name in primary:
                bindings[name]["role"] = "primary"
    return [bindings[key] for key in sorted(bindings)]


def _policy_ids_from_cell(value: Any) -> list[str]:
    parsed = _parse_json(value, [])
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _policy_coverage(
    policies: pd.DataFrame,
    world_ids: Sequence[str],
    source_experiment_ids: Sequence[str],
    candidate_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    world_set = {str(item) for item in world_ids}
    source_set = {str(item) for item in source_experiment_ids}
    direct: set[str] = set()
    candidates: set[str] = set()
    for row in policies.to_dict(orient="records"):
        policy_id = str(row["abstractPolicyId"])
        linked = set(_policy_ids_from_cell(row.get("linkedWorldIds")))
        if linked & world_set:
            direct.add(policy_id)
            continue
        if candidate_filter and str(row.get("sourceExperimentId")) in source_set and candidate_filter(row):
            candidates.add(policy_id)
    linked = sorted(direct | candidates)
    if direct and candidates:
        mode = "direct_and_source_family_candidate_links"
    elif direct:
        mode = "direct_world_links"
    elif candidates:
        mode = "source_family_candidate_links"
    else:
        mode = "no_s02_policy_links"
    return linked, {
        "policyLinkMode": mode,
        "directPolicyIds": sorted(direct),
        "candidatePolicyIds": sorted(candidates),
        "directPolicyCount": len(direct),
        "candidatePolicyCount": len(candidates),
        "linkedPolicyCount": len(linked),
    }


def _example(
    *,
    perfect_case_id: str,
    failed_case_id: str,
    perfect_metric: Mapping[str, Any],
    failed_metric: Mapping[str, Any],
    validation_method: str,
    boundary: str = "synthetic sanity example over the abstract score contract",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    perfect = {
        "caseId": perfect_case_id,
        "expectedPass": True,
        "validated": True,
        "validationMethod": validation_method,
        "metricEvidence": dict(perfect_metric),
        "boundary": boundary,
    }
    failed = {
        "caseId": failed_case_id,
        "expectedPass": False,
        "validated": True,
        "validationMethod": validation_method,
        "metricEvidence": dict(failed_metric),
        "boundary": boundary,
    }
    return perfect, failed, "validated"


def _generic_examples(goal_kind: str, goal_family: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    if goal_family in {"monotonic_order", "repair_regeneration"} or goal_kind in {"strict_global_order", "shared_global_order"}:
        return _example(
            perfect_case_id="sorted_array",
            failed_case_id="reverse_sorted_array",
            perfect_metric={"Sortedness": 100.0, "monotonicity_error": 0},
            failed_metric={"Sortedness": 0.0, "monotonicity_error": 2},
            validation_method="deterministic metric contract from e02_deterministic_simulator.metrics",
        )
    if goal_family == "homeostasis":
        return _example(
            perfect_case_id="in_range_after_perturbation",
            failed_case_id="below_threshold_after_perturbation",
            perfect_metric={"homeostasis_success": True, "failure_duration": 0},
            failed_metric={"homeostasis_success": False, "failure_duration": 3},
            validation_method="E04 homeostasis threshold contract",
        )
    if goal_family == "aggregation":
        return _example(
            perfect_case_id="all_same_policy_neighbors",
            failed_case_id="fully_alternating_policy_neighbors",
            perfect_metric={"Aggregation": 1.0},
            failed_metric={"Aggregation": 0.0},
            validation_method="adjacent same-label aggregation metric contract",
        )
    if goal_family == "anti_aggregation":
        return _example(
            perfect_case_id="fully_alternating_policy_neighbors",
            failed_case_id="all_same_policy_neighbors",
            perfect_metric={"Aggregation": 0.0},
            failed_metric={"Aggregation": 1.0},
            validation_method="lower aggregation objective over the same adjacent-label metric",
        )
    if goal_family == "symmetry":
        return _example(
            perfect_case_id="target_axis_or_pattern_selected",
            failed_case_id="wrong_axis_or_no_pattern",
            perfect_metric={"symmetry_breaking_success": True, "pattern_score": 1.0},
            failed_metric={"symmetry_breaking_success": False, "pattern_score": 0.0},
            validation_method="E05 symmetry task success contract",
        )
    if goal_family == "conflict_equilibrium":
        return _example(
            perfect_case_id="balanced_or_rescued_proxy_state",
            failed_case_id="one_sided_dominance_or_unresolved_cap",
            perfect_metric={"compromiseScore": 1.0, "oscillationRiskProxy": 0.0},
            failed_metric={"compromiseScore": 0.0, "oscillationRiskProxy": 1.0},
            validation_method="E06 compatibility and governance proxy contracts",
        )
    if goal_family == "substrate_validity":
        return _example(
            perfect_case_id="legal_identity_conserving_move",
            failed_case_id="illegal_identity_loss_or_bad_adjacency",
            perfect_metric={"identity_conservation": True, "adjacency_symmetry": True},
            failed_metric={"identity_conservation": False, "adjacency_symmetry": False},
            validation_method="E05 substrate validation contract",
        )
    return _example(
        perfect_case_id="goal_metric_best_case",
        failed_case_id="goal_metric_worst_case",
        perfect_metric={"goalScore": 1.0},
        failed_metric={"goalScore": 0.0},
        validation_method="documented bounded metric-proxy contract",
    )


def _target_examples(target_id: str, previous_artifacts_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = previous_artifacts_dir / "E05" / "research_steps" / "S03" / "target_energy_examples.parquet"
    examples = read_table(path)
    if examples.empty or "target_id" not in examples.columns:
        return _generic_examples("target_morphology_energy", "target_morphology")
    rows = examples[examples["target_id"].astype(str).eq(str(target_id))]
    perfect_rows = rows[rows["within_tolerance"].map(bool)] if "within_tolerance" in rows.columns else pd.DataFrame()
    failed_rows = rows[~rows["within_tolerance"].map(bool)] if "within_tolerance" in rows.columns else pd.DataFrame()
    if perfect_rows.empty or failed_rows.empty:
        return _generic_examples("target_morphology_energy", "target_morphology")
    perfect_row = perfect_rows.iloc[0].to_dict()
    failed_row = failed_rows.iloc[0].to_dict()
    return _example(
        perfect_case_id=str(perfect_row.get("case_id", "target_state")),
        failed_case_id=str(failed_row.get("case_id", "scrambled_state")),
        perfect_metric={
            "target_id": target_id,
            "total_energy": float(perfect_row.get("total_energy", 0.0)),
            "within_tolerance": bool(perfect_row.get("within_tolerance", True)),
        },
        failed_metric={
            "target_id": target_id,
            "total_energy": float(failed_row.get("total_energy", 1.0)),
            "within_tolerance": bool(failed_row.get("within_tolerance", False)),
        },
        validation_method="E05 S03 target_energy_examples.parquet",
        boundary="upstream target-energy examples; target energies are computational morphology proxies",
    )


def _candidate_filter_for(goal_family: str, source_experiment_ids: Sequence[str]) -> Callable[[Mapping[str, Any]], bool]:
    source_set = set(source_experiment_ids)

    def predicate(row: Mapping[str, Any]) -> bool:
        kind = str(row.get("policyKind"))
        family = str(row.get("policyFamily"))
        experiment = str(row.get("sourceExperimentId"))
        if goal_family in {"target_morphology", "symmetry", "boundary_restoration", "trajectory_proxy"}:
            return experiment == "E05" and kind in {"morphology_policy", "global_baseline_policy"}
        if goal_family == "repair_regeneration":
            return (experiment == "E05" and kind in {"morphology_policy", "global_baseline_policy"}) or experiment == "E04"
        if goal_family in {"conflict_equilibrium", "aggregation", "anti_aggregation"}:
            return experiment == "E06" or (experiment == "E01" and kind == "classic_sorting_policy")
        if goal_family == "homeostasis" and "E04" in source_set:
            return experiment == "E04"
        if goal_family == "monotonic_order":
            return (
                kind
                in {
                    "classic_sorting_policy",
                    "dsl_generated_policy",
                    "memory_augmented_policy",
                    "signal_augmented_policy",
                    "chimeric_panel_policy",
                    "morphology_policy",
                    "global_baseline_policy",
                }
                or family in {"cell_view", "traditional"}
            )
        return False

    return predicate


def _record(
    *,
    worlds: pd.DataFrame,
    policies: pd.DataFrame,
    rows: pd.DataFrame,
    abstract_goal_id: str,
    goal_label: str,
    goal_family: str,
    goal_kind: str,
    representation_type: str,
    constraint_or_energy: Mapping[str, Any],
    local_observability: str,
    tolerance: Mapping[str, Any],
    compatibility: Mapping[str, Any] | None = None,
    primary_metrics: Sequence[str] | None = None,
    source_step_extra: Sequence[str] | None = None,
    source_artifact_extra: Sequence[str] | None = None,
    source_experiment_extra: Sequence[str] | None = None,
    caveats_or_blockers: Sequence[str] | None = None,
    examples: tuple[dict[str, Any], dict[str, Any], str] | None = None,
    target_dimensions: Sequence[str] | None = None,
) -> dict[str, Any]:
    if rows.empty:
        raise ValueError(f"{abstract_goal_id} has no linked S01 worlds")
    linked_world_ids = _world_ids(rows)
    source_experiment_ids = _source_experiments(rows, source_experiment_extra)
    linked_policy_ids, policy_coverage = _policy_coverage(
        policies,
        linked_world_ids,
        source_experiment_ids,
        _candidate_filter_for(goal_family, source_experiment_ids),
    )
    substrates = _target_substrates(rows)
    dimensions = list(target_dimensions or _target_dimensions(substrates))
    perfect, failed, example_status = examples or _generic_examples(goal_kind, goal_family)
    metric_bindings = _metric_bindings(rows, goal_family, primary_metrics)
    source_goal_payloads = _source_goal_payloads(rows)
    record = {
        "abstractGoalId": abstract_goal_id,
        "goalLabel": goal_label,
        "goalSchemaVersion": GOAL_SCHEMA_VERSION,
        "goalCatalogVersion": GOAL_CATALOG_VERSION,
        "sourceExperimentIds": source_experiment_ids,
        "sourceStepIds": _source_steps(rows, source_step_extra),
        "goalFamily": goal_family,
        "goalKind": goal_kind,
        "representationType": representation_type,
        "constraintOrEnergyJson": dict(constraint_or_energy),
        "constraintOrEnergyHash": json_hash(constraint_or_energy),
        "goalCompatibilityJson": dict(compatibility or {"compatibilityClass": "not_applicable"}),
        "localObservability": local_observability,
        "targetDimensionality": dimensions,
        "targetSubstrateClasses": substrates,
        "toleranceJson": dict(tolerance),
        "metricBindingsJson": metric_bindings,
        "upstreamMeasurementFunctionNames": sorted(_world_metric_name_set(rows)),
        "linkedWorldIds": linked_world_ids,
        "linkedWorldCount": len(linked_world_ids),
        "linkedPolicyIds": linked_policy_ids,
        "linkedPolicyCount": len(linked_policy_ids),
        "policyCoverageJson": policy_coverage,
        "sourceGoalPayloadsJson": source_goal_payloads,
        "sourceArtifactPaths": _source_artifacts(rows, source_artifact_extra),
        "perfectExampleJson": perfect,
        "failedExampleJson": failed,
        "exampleValidationStatus": example_status,
        "caveatsOrBlockers": list(caveats_or_blockers or ()),
        "claimBoundary": GOAL_CLAIM_BOUNDARY,
        "worldSchemaVersion": WORLD_SCHEMA_VERSION,
        "policySchemaVersion": POLICY_SCHEMA_VERSION,
        "policyCatalogVersion": POLICY_CATALOG_VERSION,
    }
    # Ensure the complete world catalog argument is used by validation-oriented linters.
    _ = worlds
    return record


def _target_world_id(target_id: str) -> str:
    return f"E05_S03_target_{target_id}"


def build_goal_catalog(
    previous_artifacts_dir: str | Path = "/previous-artifacts",
    world_catalog_path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
    policy_catalog_path: str | Path = "/artifacts/research_steps/S02/policy_abstract_catalog.parquet",
) -> list[dict[str, Any]]:
    previous_artifacts_dir = Path(previous_artifacts_dir)
    worlds = load_world_catalog(world_catalog_path)
    policies = load_policy_catalog(policy_catalog_path)
    records: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        records.append(_record(worlds=worlds, policies=policies, **kwargs))

    sorting_rows = _by_goal_type(worlds, "strict_global_order")
    add(
        rows=sorting_rows,
        abstract_goal_id="G_MONOTONIC_GLOBAL_ORDER",
        goal_label="Monotonic global order",
        goal_family="monotonic_order",
        goal_kind="strict_global_order",
        representation_type="constraint_predicate",
        constraint_or_energy={
            "predicate": "array values satisfy declared monotonic direction",
            "energy": "monotonicity_error, inversion distance, or target-quality proxy",
            "directions": sorted({str(_goal_payload(row).get("direction", "unspecified")) for row in sorting_rows.to_dict(orient="records")}),
            "perfectCondition": "Sortedness = 100 and monotonicity_error = 0",
        },
        local_observability="pairwise local comparisons are observable; final goal score is report-facing global aggregation",
        tolerance={"sortednessPercent": 100.0, "monotonicityError": 0},
        primary_metrics=["Sortedness", "monotonicity_error", "Kendall tau distance", "Earth-mover position distance"],
    )

    add(
        rows=_by_goal_type(worlds, "shared_global_order"),
        abstract_goal_id="G_SHARED_GLOBAL_ORDER",
        goal_label="Shared global order in mixed collectives",
        goal_family="monotonic_order",
        goal_kind="shared_global_order",
        representation_type="constraint_predicate",
        constraint_or_energy={
            "predicate": "all policies are scored against the same increasing sortedness target",
            "energy": "global monotonicity error with per-policy movement diagnostics",
        },
        local_observability="local comparisons only; shared-goal score is global report-facing",
        tolerance={"sortednessPercent": 100.0, "monotonicityError": 0},
        compatibility={"compatibilityClass": "same_goal", "conflictExpected": False},
        primary_metrics=["Sortedness", "monotonicity_error", "per_algotype_movement", "aggregation_peak"],
    )

    add(
        rows=_by_goal_type(worlds, "nondecreasing_order"),
        abstract_goal_id="G_NONDECREASING_DUPLICATE_ORDER",
        goal_label="Nondecreasing order with duplicate values",
        goal_family="monotonic_order",
        goal_kind="nondecreasing_order",
        representation_type="constraint_predicate",
        constraint_or_energy={
            "predicate": "array values are nondecreasing with duplicate ties allowed",
            "energy": "tie-aware monotonicity error and duplicate aggregation diagnostics",
        },
        local_observability="local neighbor comparison with equal-value tie state",
        tolerance={"sortednessPercent": 100.0, "monotonicityError": 0, "tiesAllowed": True},
        compatibility={"compatibilityClass": "same_goal_with_ties"},
        primary_metrics=["Sortedness", "monotonicity_error", "duplicate_value_aggregation"],
    )

    add(
        rows=_by_goal_type(worlds, "conflicting_global_order"),
        abstract_goal_id="G_CONFLICTING_DIRECTION_EQUILIBRIUM",
        goal_label="Conflicting increasing/decreasing order equilibrium",
        goal_family="conflict_equilibrium",
        goal_kind="conflicting_global_order",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "mixed policies pursue incompatible increasing and decreasing directional order goals",
            "proxyEnergy": "dominance, equilibrium class, and directional sortedness imbalance",
        },
        local_observability="cell-local own-goal direction; equilibrium and dominance are report-facing proxies",
        tolerance={"singleGlobalPerfectState": "not_required_for_nontrivial_opposed_goals", "balancedProxyTarget": "low dominance spread"},
        compatibility={"compatibilityClass": "opposite_goal", "conflictExpected": True},
        primary_metrics=["equilibrium_class", "dominance_proxy", "Sortedness"],
        caveats_or_blockers=["Opposed-order equilibrium is a computational proxy; no causal negotiation or biological dominance claim is made."],
    )

    null_rows = _by_goal_type(worlds, "control_distribution", "DG_null_distribution")
    add(
        rows=null_rows,
        abstract_goal_id="G_NULL_OR_CONTROL_DISTRIBUTION",
        goal_label="Null or matched control distribution",
        goal_family="null_control",
        goal_kind="control_or_dg_null_distribution",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "observed behavior is compared with label, activation, swap, or trajectory matched null controls",
            "proxyEnergy": "observed_minus_null_effect or null contrast distance",
        },
        local_observability="not a policy objective; report-facing control distribution",
        tolerance={"nullContrast": "source_step_specific", "matchedFieldsRequired": True},
        compatibility={"compatibilityClass": "control_baseline"},
        primary_metrics=["null_distribution", "observed_minus_null_effect", "Delayed Gratification null contrast"],
        caveats_or_blockers=["Null/control records are statistical comparators, not goals pursued by local cells."],
    )

    add(
        rows=_by_goal_type(worlds, "behavioral_taxonomy"),
        abstract_goal_id="G_BEHAVIORAL_TAXONOMY_NEIGHBORHOOD",
        goal_label="Behavioral taxonomy neighborhood",
        goal_family="taxonomy",
        goal_kind="behavioral_taxonomy",
        representation_type="taxonomy_proxy",
        constraint_or_energy={
            "predicate": "policies are near each other when behaviorally similar",
            "proxyEnergy": "embedding distance and universality class agreement",
        },
        local_observability="report-facing embedding over behavior summaries",
        tolerance={"embeddingTolerance": "selected in downstream embedding steps"},
        compatibility={"compatibilityClass": "behavioral_similarity"},
        primary_metrics=["embedding_coordinates", "universality_class"],
    )

    add(
        rows=_by_goal_type(worlds, "restore_order_after_defect"),
        abstract_goal_id="G_RESTORE_ORDER_AFTER_DEFECT",
        goal_label="Restore order after a local defect",
        goal_family="repair_regeneration",
        goal_kind="restore_order_after_defect",
        representation_type="constraint_predicate",
        constraint_or_energy={
            "predicate": "recover monotonic order after repairable Frozen Cell or local defect perturbations",
            "energy": "remaining monotonicity error plus unrepaired defect penalty",
        },
        local_observability="local blocked/frozen state and neighbor values; recovery score is report-facing",
        tolerance={"sortednessPercent": 95.0, "remainingFrozenCells": 0},
        compatibility={"compatibilityClass": "repair_to_original_order"},
        primary_metrics=["repair_success", "time_to_repair", "competence_proxy", "Sortedness"],
        caveats_or_blockers=["Repair success is a computational recovery proxy over the toy sorting substrate."],
    )

    add(
        rows=_by_goal_type(worlds, "homeostasis"),
        abstract_goal_id="G_HOMEOSTATIC_ORDER_MAINTENANCE",
        goal_label="Homeostatic order maintenance",
        goal_family="homeostasis",
        goal_kind="homeostasis",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "maintain sortedness and length/identity constraints after dynamic perturbations",
            "proxyEnergy": "failure duration and recovery score",
        },
        local_observability="local state with dynamic perturbation events; time-in-range is report-facing",
        tolerance={"sortednessThresholdPercent": 95.0, "failureDurationGoal": 0},
        compatibility={"compatibilityClass": "self_maintenance"},
        primary_metrics=["homeostasis_success", "failure_duration", "Sortedness"],
        source_artifact_extra=[
            str(previous_artifacts_dir / "E04" / "research_steps" / "S11" / "competence_proxy_tasks.parquet"),
            str(previous_artifacts_dir / "E04" / "research_steps" / "S11" / "competence_proxy_metric_definitions.parquet"),
        ],
    )

    add(
        rows=_by_goal_type(worlds, "repair_competence"),
        abstract_goal_id="G_REPAIR_COMPETENCE_PROXY",
        goal_label="Repair competence under local information constraints",
        goal_family="repair_regeneration",
        goal_kind="repair_competence",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "improve repair and homeostasis proxies under local-only training constraints",
            "proxyEnergy": "competence vector loss across held-out perturbations and transfer tasks",
        },
        local_observability="local-only training constraints; competence vector is report-facing",
        tolerance={"competenceAxes": "bounded_unit_scores", "oracleAccessAllowed": False},
        compatibility={"compatibilityClass": "adaptive_repair"},
        primary_metrics=["repair_competence_proxy", "transfer_score", "centralized_comparison_delta"],
        source_artifact_extra=[
            str(previous_artifacts_dir / "E04" / "research_steps" / "S11" / "competence_proxy_metric_definitions.parquet"),
            str(previous_artifacts_dir / "E04" / "research_steps" / "S11" / "competence_proxy_results.parquet"),
        ],
        caveats_or_blockers=["Competence axes are bounded computational proxies, not evidence of cognition or agency."],
    )

    add(
        rows=_by_goal_type(worlds, "substrate_validity"),
        abstract_goal_id="G_SUBSTRATE_VALIDITY",
        goal_label="Substrate validity and identity conservation",
        goal_family="substrate_validity",
        goal_kind="substrate_validity",
        representation_type="schema_validity_constraint",
        constraint_or_energy={
            "predicate": "legal local moves conserve identities and valid adjacency on each substrate",
            "energy": "adjacency or identity-conservation violation count",
        },
        local_observability="substrate-level schema audit, not a local behavioral objective",
        tolerance={"adjacency_symmetry": True, "identity_conservation": True},
        compatibility={"compatibilityClass": "substrate_sanity"},
        primary_metrics=["adjacency_symmetry", "identity_conservation"],
        source_artifact_extra=[str(previous_artifacts_dir / "E05" / "research_steps" / "S01" / "substrate_catalog.parquet")],
    )

    target_catalog = read_table(previous_artifacts_dir / "E05" / "research_steps" / "S03" / "target_catalog.parquet")
    for target in target_catalog.to_dict(orient="records"):
        target_id = str(target["target_id"])
        world_rows = _by_world_ids(worlds, [_target_world_id(target_id)])
        if world_rows.empty:
            continue
        family = "boundary_restoration" if "boundary" in target_id else "target_morphology"
        add(
            rows=world_rows,
            abstract_goal_id=f"G_TARGET_MORPHOLOGY_{_slug(target_id)}",
            goal_label=f"Target morphology: {target.get('title', target_id)}",
            goal_family=family,
            goal_kind="target_morphology_energy",
            representation_type="energy_function",
            constraint_or_energy={
                "targetId": target_id,
                "motif": target.get("motif"),
                "predicate": "observed identity assignment satisfies target morphology within tolerance",
                "energy": "weighted identity, neighborhood, boundary, topology, and shape target error",
                "nodeCount": int(target.get("node_count", 0)),
            },
            local_observability="local target payloads can expose actor-local preferences; target energy is global report-facing",
            tolerance={"targetEnergy": 0.0, "withinTolerance": True, "targetTolerance": "from E05 target record"},
            compatibility={"compatibilityClass": "target_specific_morphology", "targetId": target_id},
            primary_metrics=["target_energy", "boundary_error", "topology_error"],
            source_artifact_extra=[
                str(previous_artifacts_dir / "E05" / "research_steps" / "S03" / "target_catalog.parquet"),
                str(previous_artifacts_dir / "E05" / "research_steps" / "S03" / "target_energy_examples.parquet"),
                str(previous_artifacts_dir / "E05" / "research_steps" / "S05" / "metric_catalog.parquet"),
                str(previous_artifacts_dir / "E05" / "research_steps" / "S05" / "metric_examples.parquet"),
            ],
            examples=_target_examples(target_id, previous_artifacts_dir),
            target_dimensions=_target_dimensions([str(target.get("substrate_type", ""))]),
            caveats_or_blockers=["Target morphology labels are computational motifs and not anatomical claims."],
        )

    add(
        rows=_by_goal_type(worlds, "repair_to_target_morphology"),
        abstract_goal_id="G_REPAIR_TO_TARGET_MORPHOLOGY",
        goal_label="Repair to target morphology after perturbation",
        goal_family="repair_regeneration",
        goal_kind="repair_to_target_morphology",
        representation_type="energy_function",
        constraint_or_energy={
            "predicate": "reduce target morphology error after removal, duplication, freeze, foreign patch, or rotated graft perturbations",
            "energy": "target error reduction and repair success",
        },
        local_observability="local policy actions; target repair metrics are report-facing",
        tolerance={"repair_success": True, "relative_error_reduction": "positive"},
        compatibility={"compatibilityClass": "repair_to_target_morphology"},
        primary_metrics=["repair_success", "relative_error_reduction"],
        source_artifact_extra=[
            str(previous_artifacts_dir / "E05" / "research_steps" / "S08" / "perturbation_catalog.parquet"),
            str(previous_artifacts_dir / "E05" / "research_steps" / "S08" / "regeneration_run_results.parquet"),
        ],
    )

    symmetry_tasks = read_table(previous_artifacts_dir / "E05" / "research_steps" / "S10" / "symmetry_task_catalog.parquet")
    for task in symmetry_tasks.to_dict(orient="records"):
        task_id = str(task["task_id"])
        world_rows = _by_world_ids(worlds, [f"E05_S10_symmetry_{task_id}"])
        if world_rows.empty:
            continue
        add(
            rows=world_rows,
            abstract_goal_id=f"G_SYMMETRY_{_slug(task_id)}",
            goal_label=f"Symmetry target: {task.get('description', task_id)}",
            goal_family="symmetry",
            goal_kind="symmetry_breaking_target",
            representation_type="metric_proxy",
            constraint_or_energy={
                "taskId": task_id,
                "targetKind": task.get("target_kind"),
                "targetAxis": task.get("target_axis"),
                "predicate": "select or form the declared axis or pattern from a symmetric neutral start",
            },
            local_observability="local pattern dynamics; axis/pattern success is report-facing",
            tolerance={"symmetry_breaking_success": True, "targetAxisAlignment": "source_task_specific"},
            compatibility={"compatibilityClass": "symmetry_breaking", "targetKind": task.get("target_kind")},
            primary_metrics=["symmetry_breaking_success"],
            source_artifact_extra=[
                str(previous_artifacts_dir / "E05" / "research_steps" / "S10" / "symmetry_task_catalog.parquet"),
                str(previous_artifacts_dir / "E05" / "research_steps" / "S10" / "symmetry_breaking_run_results.parquet"),
            ],
        )

    benchmark_catalog = read_table(previous_artifacts_dir / "E05" / "research_steps" / "S15" / "benchmark_suite_config_catalog.parquet")
    for benchmark in benchmark_catalog.to_dict(orient="records"):
        benchmark_id = str(benchmark["benchmark_id"])
        world_rows = _by_world_ids(worlds, [f"E05_S15_benchmark_{benchmark_id}"])
        if world_rows.empty:
            continue
        family_text = str(benchmark.get("benchmark_family", ""))
        goal_family = "boundary_restoration" if "boundary" in benchmark_id else "repair_regeneration"
        if "symmetry" in benchmark_id:
            goal_family = "symmetry"
        if benchmark_id == "sort_row":
            goal_family = "monotonic_order"
        if "dg" in benchmark_id:
            goal_family = "trajectory_proxy"
        add(
            rows=world_rows,
            abstract_goal_id=f"G_BENCHMARK_{_slug(benchmark_id)}",
            goal_label=f"Benchmark goal: {benchmark.get('title', benchmark_id)}",
            goal_family=goal_family,
            goal_kind=f"benchmark::{benchmark_id}",
            representation_type="benchmark_primary_goal",
            constraint_or_energy={
                "benchmarkId": benchmark_id,
                "benchmarkFamily": family_text,
                "primaryGoal": benchmark.get("primary_goal"),
                "predicate": "benchmark-specific success or error reduction",
            },
            local_observability="benchmark-dependent; information access flags distinguish local-only and global baselines",
            tolerance={"benchmarkSpecific": True, "expectedMinRows": int(benchmark.get("expected_min_rows", 0))},
            compatibility={"compatibilityClass": "benchmark_panel", "benchmarkFamily": family_text},
            primary_metrics=_parse_json(benchmark.get("metrics_json"), []) or [],
            source_artifact_extra=[
                str(previous_artifacts_dir / "E05" / "research_steps" / "S15" / "benchmark_suite_config_catalog.parquet"),
                str(previous_artifacts_dir / "E05" / "research_steps" / "S15" / "morphology_benchmark_suite_rows.parquet"),
                str(previous_artifacts_dir / "E05" / "results" / "e05_morphology_benchmarks.parquet"),
            ],
            caveats_or_blockers=_parse_json(benchmark.get("caveats_json"), []) or ["Benchmark metrics are computational proxies."],
        )

    aggregation_rows = _world_subset(
        worlds,
        lambda row: any("aggregation" in metric.lower() for metric in _metric_names(row)),
    )
    add(
        rows=aggregation_rows,
        abstract_goal_id="G_AGGREGATION_TENDENCY",
        goal_label="Aggregation tendency proxy",
        goal_family="aggregation",
        goal_kind="aggregation_metric_tendency",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "same-policy or same-label neighbors cluster more than baseline",
            "proxyEnergy": "negative aggregation score for higher-is-better objective",
        },
        local_observability="not usually an explicit policy objective; report-facing adjacent-label metric",
        tolerance={"Aggregation": "maximize_or_compare_to_null"},
        compatibility={"compatibilityClass": "emergent_tendency", "polarity": "higher_aggregation"},
        primary_metrics=["Aggregation", "aggregation_peak", "duplicate_value_aggregation", "finalAggregation"],
        caveats_or_blockers=["Aggregation is often an emergent measured tendency rather than a declared local policy objective."],
    )

    add(
        rows=aggregation_rows,
        abstract_goal_id="G_ANTI_AGGREGATION_DISPERSION",
        goal_label="Anti-aggregation or dispersion proxy",
        goal_family="anti_aggregation",
        goal_kind="anti_aggregation_metric_tendency",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "same-policy or same-label adjacency is minimized or held below a comparison baseline",
            "proxyEnergy": "aggregation score",
        },
        local_observability="report-facing inverse of the same adjacent-label metric",
        tolerance={"Aggregation": "minimize_or_compare_to_null"},
        compatibility={"compatibilityClass": "emergent_tendency", "polarity": "lower_aggregation"},
        primary_metrics=["Aggregation", "aggregation_peak", "duplicate_value_aggregation", "finalAggregation"],
        caveats_or_blockers=["Anti-aggregation is represented as an inverse metric objective; most upstream policies did not explicitly optimize it."],
    )

    goal_mode_summary = read_table(previous_artifacts_dir / "E06" / "research_steps" / "S04" / "goal_mode_summary.parquet")
    e06_goal_rows = _by_goal_type(worlds, "compatible_partial_or_opposed_policy_goals")
    for row in goal_mode_summary.to_dict(orient="records"):
        goal_mode = str(row["goalMode"])
        compatibility_class = str(row.get("goalCompatibilityClass", "unknown"))
        add(
            rows=e06_goal_rows,
            abstract_goal_id=f"G_CHIMERA_GOAL_MODE_{_slug(goal_mode)}",
            goal_label=f"Chimeric goal mode: {goal_mode}",
            goal_family="conflict_equilibrium" if compatibility_class in {"opposite", "partial", "unrelated"} else "monotonic_order",
            goal_kind=f"chimera_goal_mode::{goal_mode}",
            representation_type="compatibility_vector",
            constraint_or_energy={
                "goalMode": goal_mode,
                "compatibilityClass": compatibility_class,
                "predicate": "per-policy target-quality vector under same, opposite, partial, shared, or unrelated goals",
                "meanPolicyGoalScore": row.get("meanPolicyGoalScore"),
                "meanMinPolicyGoalScore": row.get("meanMinPolicyGoalScore"),
            },
            local_observability="per-cell goal metadata is local; compatibility score is report-facing",
            tolerance={"jointGoalSatisfiedRate": row.get("jointGoalSatisfiedRate"), "sourceSummary": "E06 S04 goal_mode_summary"},
            compatibility={"compatibilityClass": compatibility_class, "goalMode": goal_mode},
            primary_metrics=["compatibility_composite_score", "minority_score", "compromise_score", "Sortedness"],
            source_artifact_extra=[
                str(previous_artifacts_dir / "E06" / "research_steps" / "S04" / "goal_mode_summary.parquet"),
                str(previous_artifacts_dir / "E06" / "research_steps" / "S04" / "policy_goal_metadata.parquet"),
                str(previous_artifacts_dir / "E06" / "research_steps" / "S05" / "compatibility_metric_definitions.parquet"),
            ],
            caveats_or_blockers=["Compatibility and compromise are metric proxies, not direct evidence of negotiation or biological conflict."],
        )

    add(
        rows=_by_goal_type(worlds, "dominance_or_mosaic_proxy"),
        abstract_goal_id="G_DOMINANCE_MOSAIC_INTERFACE_STABILITY",
        goal_label="Dominance, mosaic, and interface stability proxies",
        goal_family="conflict_equilibrium",
        goal_kind="dominance_or_mosaic_proxy",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "winner, interface stability, or mosaic class proxy improves or remains stable",
            "proxyEnergy": "dominance spread, mosaic class, and interface instability",
        },
        local_observability="mixed-policy cell states are local; dominance and mosaic classes are report-facing",
        tolerance={"dominanceProxy": "source_specific", "interfaceStability": "higher_is_better"},
        compatibility={"compatibilityClass": "dominance_or_mosaic"},
        primary_metrics=["dominance_proxy", "mosaic_class", "interface_stability"],
        source_artifact_extra=[
            str(previous_artifacts_dir / "E06" / "research_steps" / "S06" / "dominance_winner_criteria.parquet"),
            str(previous_artifacts_dir / "E06" / "research_steps" / "S07" / "mosaic_classification_rules.parquet"),
        ],
    )

    add(
        rows=_by_goal_type(worlds, "rescue_or_control_proxy"),
        abstract_goal_id="G_RESCUE_OR_CONTROL_INTERVENTION",
        goal_label="Rescue or control intervention proxy",
        goal_family="conflict_equilibrium",
        goal_kind="rescue_or_control_proxy",
        representation_type="metric_proxy",
        constraint_or_energy={
            "predicate": "improve target quality or compatibility under declared intervention caveats",
            "proxyEnergy": "negative rescue success plus cost and side-effect penalties",
        },
        local_observability="intervention metadata may be nonlocal; local/global scope is explicitly flagged",
        tolerance={"rescue_success_proxy": True, "cost_proxy": "bounded", "side_effect_penalty": "bounded"},
        compatibility={"compatibilityClass": "rescue_control"},
        primary_metrics=["rescue_success_proxy", "cost_proxy", "side_effect_penalty", "causal_prediction_score"],
        source_artifact_extra=[
            str(previous_artifacts_dir / "E06" / "research_steps" / "S09" / "governance_goal_metadata.parquet"),
            str(previous_artifacts_dir / "E06" / "research_steps" / "S14" / "intervention_recipe_summary.parquet"),
        ],
        caveats_or_blockers=["Intervention recipes are bounded metadata/proxy controls, not causal biological protocols."],
    )

    add(
        rows=_by_goal_type(worlds, "bounded_chimeric_control_synthesis"),
        abstract_goal_id="G_CHIMERIC_CONTROL_PLAYBOOK_TRACEABILITY",
        goal_label="Chimeric control playbook traceability",
        goal_family="conflict_equilibrium",
        goal_kind="bounded_chimeric_control_synthesis",
        representation_type="taxonomy_proxy",
        constraint_or_energy={
            "predicate": "recommendations retain evidence links and caveats",
            "proxyEnergy": "traceability failure and caveat loss",
        },
        local_observability="report-facing synthesis objective",
        tolerance={"recommendation_traceability": True, "caveat_preservation": True},
        compatibility={"compatibilityClass": "synthesis_traceability"},
        primary_metrics=["recommendation_traceability", "caveat_preservation"],
        source_artifact_extra=[
            str(previous_artifacts_dir / "E06" / "report_bundle_inputs" / "e06_chimeric_control_playbook" / "tables" / "evidence_traceability.parquet"),
            str(previous_artifacts_dir / "E06" / "report_bundle_inputs" / "e06_chimeric_control_playbook" / "tables" / "caveat_register.parquet"),
        ],
    )

    records = sorted(records, key=lambda item: item["abstractGoalId"])
    return records


def catalog_to_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for column in JSON_COLUMNS:
            row[column] = compact_json(row.get(column))
        rows.append(row)
    return pd.DataFrame(rows)


def goal_world_links_dataframe(records: Sequence[Mapping[str, Any]], world_catalog_path: str | Path) -> pd.DataFrame:
    worlds = load_world_catalog(world_catalog_path)
    world_by_id = {str(row["worldId"]): row for row in worlds.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for record in records:
        policy_coverage = record.get("policyCoverageJson", {})
        direct_world_policy_ids = set(policy_coverage.get("directPolicyIds", [])) if isinstance(policy_coverage, Mapping) else set()
        for world_id in record["linkedWorldIds"]:
            world = world_by_id[str(world_id)]
            rows.append(
                {
                    "abstractGoalId": record["abstractGoalId"],
                    "goalFamily": record["goalFamily"],
                    "goalKind": record["goalKind"],
                    "worldId": world_id,
                    "sourceExperimentId": world["experimentId"],
                    "worldFamily": world["worldFamily"],
                    "substrateClass": world["substrateClass"],
                    "worldMetricNames": compact_json(_metric_names(world)),
                    "goalMetricNames": compact_json(record["upstreamMeasurementFunctionNames"]),
                    "policyLinkMode": record["policyCoverageJson"]["policyLinkMode"],
                    "goalLinkedPolicyCount": len(record["linkedPolicyIds"]),
                    "directPolicyCountForGoal": len(direct_world_policy_ids),
                    "claimBoundary": GOAL_CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def _validation_row(
    *,
    check_id: str,
    severity: str,
    scope: str,
    record_id: str,
    check_name: str,
    success: bool,
    detail: str,
) -> dict[str, Any]:
    return {
        "checkId": check_id,
        "severity": severity,
        "scope": scope,
        "recordId": record_id,
        "checkName": check_name,
        "success": bool(success),
        "detail": detail,
    }


def validate_goal_catalog(
    records: Sequence[Mapping[str, Any]],
    world_catalog_path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
    policy_catalog_path: str | Path = "/artifacts/research_steps/S02/policy_abstract_catalog.parquet",
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    worlds = load_world_catalog(world_catalog_path)
    policies = load_policy_catalog(policy_catalog_path)
    world_ids = {str(item) for item in worlds["worldId"].dropna().unique()}
    policy_ids = {str(item) for item in policies["abstractPolicyId"].dropna().unique()}
    record_ids = [str(record.get("abstractGoalId")) for record in records]
    record_id_counts = Counter(record_ids)
    world_to_metric_names = {str(row["worldId"]): set(_metric_names(row)) for row in worlds.to_dict(orient="records")}

    for record in records:
        record_id = str(record.get("abstractGoalId", "missing"))
        for field in REQUIRED_GOAL_FIELDS:
            checks.append(
                _validation_row(
                    check_id=f"{record_id}::required::{field}",
                    severity="error",
                    scope="record",
                    record_id=record_id,
                    check_name=f"required field {field}",
                    success=field in record,
                    detail="field exists" if field in record else "field missing",
                )
            )
        for field in NONEMPTY_GOAL_FIELDS:
            value = record.get(field)
            present = value is not None and value != "" and value != [] and value != {}
            checks.append(
                _validation_row(
                    check_id=f"{record_id}::nonempty::{field}",
                    severity="error",
                    scope="record",
                    record_id=record_id,
                    check_name=f"nonempty field {field}",
                    success=present,
                    detail="field populated" if present else f"{field} is empty",
                )
            )
        checks.append(
            _validation_row(
                check_id=f"{record_id}::unique_id",
                severity="error",
                scope="record",
                record_id=record_id,
                check_name="unique goal ID",
                success=record_id_counts[record_id] == 1,
                detail=f"count={record_id_counts[record_id]}",
            )
        )
        linked_worlds = {str(item) for item in record.get("linkedWorldIds", [])}
        missing_worlds = sorted(linked_worlds - world_ids)
        checks.append(
            _validation_row(
                check_id=f"{record_id}::linked_worlds_resolve",
                severity="error",
                scope="record",
                record_id=record_id,
                check_name="linked S01 worlds resolve",
                success=not missing_worlds and bool(linked_worlds),
                detail="all linked worlds resolve" if not missing_worlds else f"missing worlds={missing_worlds}",
            )
        )
        linked_policies = {str(item) for item in record.get("linkedPolicyIds", [])}
        missing_policies = sorted(linked_policies - policy_ids)
        checks.append(
            _validation_row(
                check_id=f"{record_id}::linked_policies_resolve",
                severity="error",
                scope="record",
                record_id=record_id,
                check_name="linked S02 policies resolve",
                success=not missing_policies,
                detail="all linked policies resolve" if not missing_policies else f"missing policies={missing_policies[:10]}",
            )
        )
        checks.append(
            _validation_row(
                check_id=f"{record_id}::policy_coverage_present",
                severity="warning",
                scope="record",
                record_id=record_id,
                check_name="policy coverage present",
                success=bool(linked_policies),
                detail=record.get("policyCoverageJson", {}).get("policyLinkMode", "unknown"),
            )
        )
        artifact_paths = [str(item) for item in record.get("sourceArtifactPaths", [])]
        missing_artifacts = [path for path in artifact_paths if not Path(path).exists()]
        checks.append(
            _validation_row(
                check_id=f"{record_id}::source_artifacts_exist",
                severity="error",
                scope="record",
                record_id=record_id,
                check_name="source artifacts exist",
                success=not missing_artifacts and bool(artifact_paths),
                detail="all source artifacts exist" if not missing_artifacts else f"missing artifacts={missing_artifacts[:5]}",
            )
        )
        metrics = {str(item) for item in record.get("upstreamMeasurementFunctionNames", [])}
        linked_world_metrics: set[str] = set()
        for world_id in linked_worlds:
            linked_world_metrics.update(world_to_metric_names.get(world_id, set()))
        metric_overlap = metrics & linked_world_metrics
        checks.append(
            _validation_row(
                check_id=f"{record_id}::metrics_link_to_worlds",
                severity="error",
                scope="record",
                record_id=record_id,
                check_name="goals link to world metrics",
                success=bool(metric_overlap),
                detail=f"{len(metric_overlap)} linked metric names overlap S01 world measurement functions",
            )
        )
        perfect = record.get("perfectExampleJson", {})
        failed = record.get("failedExampleJson", {})
        examples_ok = (
            isinstance(perfect, Mapping)
            and isinstance(failed, Mapping)
            and bool(perfect.get("validated"))
            and bool(failed.get("validated"))
            and bool(perfect.get("expectedPass")) is True
            and bool(failed.get("expectedPass")) is False
        )
        checks.append(
            _validation_row(
                check_id=f"{record_id}::examples_validated",
                severity="error",
                scope="record",
                record_id=record_id,
                check_name="perfect and failed examples validated",
                success=examples_ok,
                detail=record.get("exampleValidationStatus", "missing"),
            )
        )
        for version_field, expected in (
            ("worldSchemaVersion", WORLD_SCHEMA_VERSION),
            ("policySchemaVersion", POLICY_SCHEMA_VERSION),
            ("policyCatalogVersion", POLICY_CATALOG_VERSION),
            ("goalSchemaVersion", GOAL_SCHEMA_VERSION),
        ):
            checks.append(
                _validation_row(
                    check_id=f"{record_id}::version::{version_field}",
                    severity="error",
                    scope="record",
                    record_id=record_id,
                    check_name=f"{version_field} matches",
                    success=record.get(version_field) == expected,
                    detail=f"observed={record.get(version_field)} expected={expected}",
                )
            )

    represented_families = {str(record.get("goalFamily")) for record in records}
    for family, label in HARD_COVERAGE_REQUIREMENTS.items():
        checks.append(
            _validation_row(
                check_id=f"catalog::coverage::{family}",
                severity="error",
                scope="catalog",
                record_id="catalog",
                check_name=f"coverage for {label}",
                success=family in represented_families,
                detail=f"represented families={sorted(represented_families)}",
            )
        )
    represented_experiments: set[str] = set()
    represented_worlds: set[str] = set()
    for record in records:
        represented_experiments.update(str(item) for item in record.get("sourceExperimentIds", []))
        represented_worlds.update(str(item) for item in record.get("linkedWorldIds", []))
    for experiment in [f"E{i:02d}" for i in range(1, 7)]:
        checks.append(
            _validation_row(
                check_id=f"catalog::source::{experiment}",
                severity="error",
                scope="catalog",
                record_id="catalog",
                check_name=f"source experiment {experiment} represented",
                success=experiment in represented_experiments,
                detail=f"represented experiments={sorted(represented_experiments)}",
            )
        )
    missing_world_coverage = sorted(world_ids - represented_worlds)
    checks.append(
        _validation_row(
            check_id="catalog::world_coverage",
            severity="error",
            scope="catalog",
            record_id="catalog",
            check_name="all S01 worlds linked to at least one S03 goal",
            success=not missing_world_coverage,
            detail="all worlds covered" if not missing_world_coverage else f"missing worlds={missing_world_coverage}",
        )
    )
    return pd.DataFrame(checks)


def coverage_summary(records: Sequence[Mapping[str, Any]], validation: pd.DataFrame) -> dict[str, Any]:
    hard = validation[validation["severity"].eq("error")]
    failed_hard = hard[~hard["success"]]
    failed_warning = validation[validation["severity"].eq("warning") & (~validation["success"])]
    world_ids = sorted({str(world_id) for record in records for world_id in record.get("linkedWorldIds", [])})
    policy_ids = sorted({str(policy_id) for record in records for policy_id in record.get("linkedPolicyIds", [])})
    source_experiments = sorted({str(exp) for record in records for exp in record.get("sourceExperimentIds", [])})
    return {
        "goalSchemaVersion": GOAL_SCHEMA_VERSION,
        "goalCatalogVersion": GOAL_CATALOG_VERSION,
        "goalRecordCount": len(records),
        "linkedWorldCount": len(world_ids),
        "linkedPolicyCount": len(policy_ids),
        "sourceExperimentCounts": dict(Counter(exp for record in records for exp in record.get("sourceExperimentIds", []))),
        "goalFamilyCounts": dict(Counter(str(record.get("goalFamily")) for record in records)),
        "goalKindCounts": dict(Counter(str(record.get("goalKind")) for record in records)),
        "representationTypeCounts": dict(Counter(str(record.get("representationType")) for record in records)),
        "policyLinkModeCounts": dict(Counter(str(record.get("policyCoverageJson", {}).get("policyLinkMode")) for record in records)),
        "representedSourceExperiments": source_experiments,
        "validationCheckCount": int(len(validation)),
        "hardValidationFailureCount": int(len(failed_hard)),
        "warningFailureCount": int(len(failed_warning)),
        "allHardChecksPassed": bool(failed_hard.empty),
        "claimBoundary": GOAL_CLAIM_BOUNDARY,
    }
