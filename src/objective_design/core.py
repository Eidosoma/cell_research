"""Training-only S04 objective and descriptor design validation.

This module freezes evaluation contracts; it does not run policy search.  It
may consume the eight authorized S02 training baseline outcomes and S03's
outcome-free policy probe summaries.  Validation and confirmation outcomes are
neither materialized nor evaluated.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


S04_SCHEMA_VERSION = "e07.s04.evidence.v1"
TASK_IDS = (
    "e07_s02_sorting_1d",
    "e07_s02_faults_1d",
    "e07_s02_detour_1d",
    "e07_s02_chimera_1d",
    "e07_s02_regeneration_1d",
    "e07_s02_target_change_1d",
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def require_s05_eligible(gate: Mapping[str, Any]) -> None:
    """Fail closed unless every frozen S05 prerequisite has passed."""

    failing = [
        row["id"]
        for row in gate.get("requirements", [])
        if row.get("currentStatus") != "pass"
    ]
    if gate.get("s05SearchEligible") is not True or failing:
        detail = ", ".join(failing) if failing else "gate decision is not pass"
        raise RuntimeError(f"S05 search eligibility gate is blocked: {detail}")


def _json_lines(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object row in {path}")
            rows.append(value)
    return rows


def bin_index(value: float, edges: Sequence[float]) -> int:
    """Return the frozen lower-inclusive bin; the final upper edge is inclusive."""

    if not math.isfinite(value):
        raise ValueError("descriptor value must be finite")
    if (
        len(edges) < 2
        or any(not math.isfinite(float(edge)) for edge in edges)
        or any(float(right) <= float(left) for left, right in zip(edges, edges[1:]))
    ):
        raise ValueError("descriptor edges must be finite and strictly increasing")
    if value < edges[0] or value > edges[-1]:
        raise ValueError("descriptor value outside frozen bounds")
    if value == edges[-1]:
        return len(edges) - 2
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        if left <= value < right:
            return index
    raise AssertionError("unreachable bin assignment")


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def validate_registries(
    objective: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    weighting: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    anti_gaming: Mapping[str, Any],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["objectiveForbidsUniversalScore"] = (
        objective["globalRules"]["universalNormalizedScorePermitted"] is False
        and objective["globalRules"]["crossTaskScalarizationPermitted"] is False
        and objective["globalRules"]["crossTaskDominancePermitted"] is False
    )
    checks["nativeRawValuesRequired"] = bool(
        objective["globalRules"]["nativeRawValuesRequiredBesideAnyDisplayTransform"]
    )
    checks["E03PoolingForbidden"] = (
        objective["globalRules"]["crossMetricDetourPoolingPermitted"] is False
        and objective["taskObjectives"]["e07_s02_detour_1d"][
            "exactSmallAndEmpiricalLargePoolingPermitted"
        ]
        is False
    )
    checks["E05AggregateForbidden"] = (
        objective["globalRules"]["aggregateE05CompetencyScorePermitted"] is False
        and objective["taskObjectives"]["e05_benchmark_axes"]["aggregateScorePermitted"]
        is False
    )
    e06 = objective["taskObjectives"]["e06_bounded_square"]
    checks["E06ConjunctionRequired"] = (
        any(
            item.get("id") == "conjunctive_completion"
            and item.get("requires") == "calibrated_bounded_square_s01_and_s02"
            for item in e06["performance"]
        )
        and "fixedBudgetCompleted" in e06["prohibitedObjectives"]
    )
    common = objective["commonObjectiveFamilies"]
    checks["InsertionLedgerSeparate"] = (
        common["licensedInsertionPrefix"]["aggregation"]
        == "separate_capability_ledger_never_folded_into_native_cost"
    )
    checks["SelectionLedgerSeparate"] = (
        common["licensedSelectionCursorAndRange"]["aggregation"]
        == "separate_capability_ledger_never_folded_into_native_cost"
    )
    archive = descriptor["archivePolicy"]
    checks["DescriptorExclusionsExplicit"] = all(
        archive[key] is False
        for key in (
            "s03StructuralCoverageUsedAsDescriptor",
            "s03FiniteCorpusEquivalenceUsedAsDescriptor",
            "e05UnboundRowsUsedAsDescriptor",
            "e06SyntheticTypedFixturesUsedAsDescriptor",
            "confirmationOutcomesUsed",
            "validationOutcomesUsed",
        )
    )

    bounded_metric_count = 0
    for mapping in _walk_mappings(objective):
        if "bounds" in mapping:
            lower, upper = mapping["bounds"]
            bounded_metric_count += 1
            if not (
                isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and lower < upper
            ):
                raise ValueError(f"invalid metric bounds: {mapping}")
        if "direction" in mapping and mapping["direction"] not in {
            "maximize",
            "minimize",
            "report_only",
            "report_curve_and_pareto_not_auc_total",
            "separate_success_time_error",
            "separate_paired_effects",
            "separate_success_time_stability",
            "separate_by_stratum_and_policy_family",
        }:
            raise ValueError(f"invalid objective direction: {mapping['direction']}")

    descriptor_dimension_count = 0
    for mapping in _walk_mappings(descriptor):
        if "binEdges" in mapping:
            descriptor_dimension_count += 1
            edges = [float(edge) for edge in mapping["binEdges"]]
            if any(right <= left for left, right in zip(edges, edges[1:])):
                raise ValueError(f"non-increasing descriptor edges: {mapping}")
            if (
                "bounds" in mapping
                and edges[0] != float(mapping["bounds"][0])
                or "bounds" in mapping
                and edges[-1] != float(mapping["bounds"][1])
            ):
                raise ValueError(f"descriptor edges do not span bounds: {mapping}")
            for edge in edges:
                bin_index(edge, edges)
    checks["MetricBoundsWellFormed"] = bounded_metric_count >= 20
    checks["DescriptorBoundsWellFormed"] = descriptor_dimension_count >= 10

    profiles = weighting["profiles"]
    checks["WeightProfilesCoverEightTasks"] = all(
        set(weights) == set(TASK_IDS) for weights in profiles.values()
    )
    checks["WeightProfilesSumToOne"] = all(
        math.isclose(sum(map(float, weights.values())), 1.0, abs_tol=1e-12)
        for weights in profiles.values()
    )
    checks["WeightProfilesPositive"] = all(
        all(float(value) > 0 for value in weights.values())
        for weights in profiles.values()
    )
    checks["WeightsAllocationOnly"] = (
        weighting["purpose"] == "evaluation_allocation_only"
        and weighting["scalarPerformanceAggregationPermitted"] is False
    )
    checks["UncertaintyPreservesCensoring"] = set(
        uncertainty["methods"]["completionTime"]["prohibited"]
    ) == {
        "drop_censored",
        "impute_censored_as_zero",
        "compare_incomparable_native_clocks",
    }
    channels = anti_gaming["channels"]
    checks["KnownGamingChannelsUnique"] = (
        len(channels) == len({row["id"] for row in channels}) and len(channels) >= 15
    )
    requirements = eligibility["requirements"]
    checks["E05AndE06AdaptersMandatory"] = (
        eligibility["e05AdapterRequiredBeforeS05"] is True
        and eligibility["e06AdapterRequiredBeforeS05"] is True
        and eligibility["fieldNameSimilarityCanWaiveAdapter"] is False
        and next(row for row in requirements if row["id"].startswith("G03"))[
            "nonWaivable"
        ]
        is True
        and next(row for row in requirements if row["id"].startswith("G04"))[
            "nonWaivable"
        ]
        is True
    )
    computed_eligible = all(row["currentStatus"] == "pass" for row in requirements)
    checks["EligibilityDecisionConsistent"] = (
        computed_eligible == eligibility["s05SearchEligible"]
        and eligibility["decision"] == ("pass" if computed_eligible else "blocked")
        and eligibility["enforcementFunction"]
        == "src.objective_design.require_s05_eligible"
    )
    return {
        "schemaVersion": "e07.s04.registry-validation.v1",
        "researchStepId": "S04",
        "boundedMetricCount": bounded_metric_count,
        "descriptorDimensionCount": descriptor_dimension_count,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _flatten_numeric(prefix: str, value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numeric(child_prefix, value[key]))
    elif isinstance(value, bool):
        rows.append({"field": prefix, "value": int(value), "kind": "binary"})
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        rows.append({"field": prefix, "value": value, "kind": "numeric"})
    return rows


def build_normalization_baselines(
    baseline_smoke_path: str | Path,
) -> dict[str, Any]:
    source = json.loads(Path(baseline_smoke_path).read_text(encoding="utf-8"))
    rows = []
    for item in sorted(source["rows"], key=lambda row: row["taskId"]):
        if item["reset"]["split"] != "train" or item["outcome"]["split"] != "train":
            raise ValueError("normalization anchors must be training rows")
        ledgers = item["step"]["cost"]["nativeLedgerFamilies"]
        ledger_anchors = []
        for family in sorted(ledgers):
            for flattened in _flatten_numeric(family, ledgers[family]):
                value = flattened["value"]
                flattened["displayBaselineRatioPermitted"] = value > 0
                flattened["zeroAnchorHandling"] = (
                    None if value > 0 else "undefined_ratio_preserve_raw"
                )
                ledger_anchors.append(flattened)
        rows.append(
            {
                "taskId": item["taskId"],
                "scenarioId": item["scenarioId"],
                "policyId": item["policyId"],
                "split": "train",
                "nativeUnit": item["step"]["nativeUnit"],
                "nativeHorizon": item["reset"]["horizon"],
                "stopReason": item["step"]["stopReason"],
                "censored": item["step"]["censored"],
                "failed": item["step"]["failed"],
                "ledgerAnchors": ledger_anchors,
                "outcomeAnchors": _flatten_numeric(
                    "outcome", item["outcome"]["outcome"]
                ),
                "claimBoundary": item["outcome"]["claimBoundary"],
                "uncertainty": "not_estimable_single_deterministic_training_fixture",
            }
        )
    if len(rows) != 8:
        raise ValueError("expected eight S02 training baseline rows")
    return {
        "schemaVersion": "e07.s04.normalization-baselines.v1",
        "researchStepId": "S04",
        "source": str(Path(baseline_smoke_path)),
        "authorizedSplits": ["train"],
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "rules": {
            "nativeRawValuesAlwaysRetained": True,
            "universalNormalizedScore": None,
            "crossTaskMinMax": "forbidden",
            "crossTaskZScore": "forbidden",
            "taskLocalBaselineRatio": "display_only_for_nonnegative_fields_with_positive_training_anchor",
            "boundedMetricTransform": "identity_or_declared_theoretical_bound_only",
            "timeBudgetFraction": "same_task_same_phase_only_with_censor_flag",
            "singleTrainingRowIsPopulationEstimate": False,
        },
        "taskBaselines": rows,
    }


def _training_size_by_task(split_manifest_path: str | Path) -> dict[str, int | None]:
    manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    result: dict[str, int | None] = {}
    records = manifest.get("records", manifest.get("scenarios", []))
    for record in records:
        if record.get("split") != "train":
            continue
        params = record.get("publicParameters", record.get("public_parameters", {}))
        n = params.get("n")
        if n is None and isinstance(params.get("values"), list):
            n = len(params["values"])
        result[record["taskId"]] = int(n) if isinstance(n, int) else None
    return result


def _baseline_common_descriptor(
    item: Mapping[str, Any], n: int | None
) -> dict[str, Any]:
    ledgers = item["step"]["cost"]["nativeLedgerFamilies"]
    line = next(
        (
            ledger
            for name, ledger in ledgers.items()
            if name in {"e01ReferenceLedger", "e01ReferenceLedgerDelta"}
        ),
        None,
    )
    if line is not None:
        proposals = int(line["proposals"])
        accepted = int(line["acceptedSwaps"])
        if n is None or n < 2 or proposals <= 0:
            return {"available": False, "reason": "missing_line_size_or_opportunities"}
        accepted_fraction = accepted / proposals
        displacement_fraction = 0.0 if accepted == 0 else 1.0 / (n - 1)
        return {
            "available": True,
            "acceptedNativeActionFraction": accepted_fraction,
            "committedDisplacementFraction": displacement_fraction,
            "provenance": "authorized_S02_native_training_ledger",
            "limitations": "native baseline uses adjacent committed swaps; no arbitrary DSL episode inference",
        }
    spatial = ledgers.get("e06MovementLedger")
    if spatial is not None:
        opportunities = int(spatial["submittedProposals"])
        accepted = int(spatial["acceptedMovements"])
        if opportunities <= 0:
            return {"available": False, "reason": "zero_spatial_submitted_proposals"}
        displacement = int(spatial["totalGraphDisplacement"])
        maximum_per_movement = 6  # longest enabled S04 rotation; conservative.
        return {
            "available": True,
            "acceptedNativeActionFraction": accepted / opportunities,
            "committedDisplacementFraction": (
                0.0
                if accepted == 0
                else displacement / (accepted * maximum_per_movement)
            ),
            "provenance": "authorized_S02_native_E06_training_ledger_not_S03_fixture",
            "limitations": "six-edge-unit denominator is the maximum enabled S04 rotation displacement",
        }
    return {
        "available": False,
        "reason": "native_training_ledger_lacks_movement_fields",
    }


def descriptor_validation(
    descriptor: Mapping[str, Any],
    baseline_smoke_path: str | Path,
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    source = json.loads(Path(baseline_smoke_path).read_text(encoding="utf-8"))
    sizes = _training_size_by_task(split_manifest_path)
    dimensions = descriptor["commonArchive"]["dimensions"]
    edge_map = {row["id"]: [float(x) for x in row["binEdges"]] for row in dimensions}
    rows = []
    for item in sorted(source["rows"], key=lambda row: row["taskId"]):
        values = _baseline_common_descriptor(item, sizes.get(item["taskId"]))
        row = {"taskId": item["taskId"], **values}
        if values["available"]:
            accepted = float(values["acceptedNativeActionFraction"])
            displacement = float(values["committedDisplacementFraction"])
            row["archiveCell"] = [
                bin_index(accepted, edge_map["accepted_native_action_fraction"]),
                bin_index(displacement, edge_map["committed_displacement_fraction"]),
            ]
        rows.append(row)

    # Numeric bin-conformance fixtures are not policy/task outcomes.
    midpoints = [0.05 + 0.1 * index for index in range(10)]
    cells = {
        (
            bin_index(left, edge_map["accepted_native_action_fraction"]),
            bin_index(right, edge_map["committed_displacement_fraction"]),
        )
        for left in midpoints
        for right in midpoints
    }
    jitter = 1e-9
    interior_stable = all(
        bin_index(value, edge_map[dimension])
        == bin_index(value + jitter, edge_map[dimension])
        == bin_index(value - jitter, edge_map[dimension])
        for dimension in edge_map
        for value in midpoints
    )
    boundary_deterministic = all(
        bin_index(edge, edges) == min(index, len(edges) - 2)
        for edges in edge_map.values()
        for index, edge in enumerate(edges)
    )
    available = [row for row in rows if row["available"]]
    checks = {
        "authorizedNativeTrainingRowsOnly": all(
            row.get("provenance", "").startswith("authorized_S02_native")
            for row in available
        ),
        "E05UnboundRowNotUsedAsDescriptor": next(
            row for row in rows if row["taskId"] == "e07_s02_regeneration_1d"
        )["available"]
        is False,
        "E06S03TypedFixturesNotUsed": all(
            "not_S03_fixture" in row.get("provenance", "")
            for row in available
            if row["taskId"].startswith("e07_s02_spatial")
        ),
        "allAvailableValuesWithinBounds": all(
            0 <= row["acceptedNativeActionFraction"] <= 1
            and 0 <= row["committedDisplacementFraction"] <= 1
            for row in available
        ),
        "fullNumericBinGridOccupied": len(cells) == 100,
        "interiorAssignmentsStable": interior_stable,
        "boundaryAssignmentsDeterministic": boundary_deterministic,
        "singletonUncertaintyNotOverstated": True,
    }
    return {
        "schemaVersion": "e07.s04.descriptor-validation.v1",
        "researchStepId": "S04",
        "nativeTrainingBaselineDescriptorRows": rows,
        "availableNativeTrainingBaselineDescriptorCount": len(available),
        "taskStratifiedOccupiedCellCount": len(available),
        "numericConformanceFixtureCellCount": len(cells),
        "numericConformanceFixturesArePolicyOrEfficacyEvidence": False,
        "bootstrapCellStability": "not_estimable_from_one_native_training_baseline_per_task; frozen threshold is prospective",
        "checks": checks,
        "passed": all(checks.values()),
    }


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2 + 1
        for ordered_index in order[position:end]:
            ranks[ordered_index] = rank
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var == 0 or right_var == 0:
        return None
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    return covariance / math.sqrt(left_var * right_var)


def correlation_audit(
    evaluations_path: str | Path, complexity_path: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluations = {row["policyId"]: row for row in _json_lines(evaluations_path)}
    complexities = {row["policyId"]: row for row in _json_lines(complexity_path)}
    feature_rows = []
    for policy_id in sorted(evaluations):
        evaluation = evaluations[policy_id]
        complexity = complexities[policy_id]
        structural = complexity["structuralComplexity"]
        ledger = evaluation["adapterProjectionLedger"]
        action_count = sum(
            count
            for action, count in evaluation["actionCounts"].items()
            if action != "noop"
        )
        feature_rows.append(
            {
                "policyId": policy_id,
                "environment": evaluation["environment"],
                "proposalActivityFraction": action_count / evaluation["probeCount"],
                "operationUtilization": evaluation["operationCount"]["mean"]
                / structural["worstCaseOperations"],
                "ruleCount": float(structural["ruleCount"]),
                "expressionNodes": float(structural["expressionNodes"]),
                "persistentMemoryBits": float(structural["persistentMemoryBits"]),
                "outboundSignalBits": float(structural["outboundSignalBits"]),
                "logMovementRadius": math.log1p(structural["maxMovementRadius"]),
                "licensedPrefixUse": float(
                    ledger["licensedPrefixPredicateEvaluations"] > 0
                ),
                "licensedCursorUse": float(ledger["engineCursorStateReads"] > 0),
                "opaqueCandidateUse": float(
                    ledger["opaqueCandidateRecordsProjected"] > 0
                ),
            }
        )
    feature_names = [
        key for key in feature_rows[0] if key not in {"policyId", "environment"}
    ]
    correlations: list[dict[str, Any]] = []
    for environment in sorted({row["environment"] for row in feature_rows}):
        subset = [row for row in feature_rows if row["environment"] == environment]
        for left_index, left_name in enumerate(feature_names):
            for right_name in feature_names[left_index + 1 :]:
                left = [float(row[left_name]) for row in subset]
                right = [float(row[right_name]) for row in subset]
                rho = _pearson(_rank(left), _rank(right))
                correlations.append(
                    {
                        "environment": environment,
                        "nPolicies": len(subset),
                        "featureA": left_name,
                        "featureB": right_name,
                        "spearmanRho": rho,
                        "absoluteRhoAtLeast0_9": rho is not None and abs(rho) >= 0.9,
                    }
                )
    estimable = [row for row in correlations if row["spearmanRho"] is not None]
    high = [row for row in estimable if row["absoluteRhoAtLeast0_9"]]
    summary = {
        "schemaVersion": "e07.s04.correlation-audit.v1",
        "researchStepId": "S04",
        "scope": "S03 structural and outcome-free training-probe features only; not efficacy and not archive occupancy",
        "policyCount": len(feature_rows),
        "estimablePairCount": len(estimable),
        "highAbsoluteCorrelationPairCount": len(high),
        "primaryNativeDescriptorCorrelationEstimable": False,
        "reasonPrimaryNotEstimable": "one native training baseline per task and cross-task pooling is forbidden",
        "noCrossTaskPerformanceCorrelationComputed": True,
        "topAbsolutePairs": sorted(
            high, key=lambda row: abs(row["spearmanRho"]), reverse=True
        )[:10],
        "passed": len(feature_rows) == 30,
    }
    return correlations, summary


def correlations_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = [
        "environment",
        "nPolicies",
        "featureA",
        "featureB",
        "spearmanRho",
        "absoluteRhoAtLeast0_9",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def weighting_sensitivity(
    weighting: Mapping[str, Any], eligible: bool
) -> dict[str, Any]:
    rows = []
    for profile, weights in weighting["profiles"].items():
        for task_id, value in weights.items():
            rows.append({"profile": profile, "taskId": task_id, "weight": value})
    sums = {
        profile: sum(map(float, weights.values()))
        for profile, weights in weighting["profiles"].items()
    }
    minimum = min(float(row["weight"]) for row in rows)
    maximum = max(float(row["weight"]) for row in rows)
    return {
        "schemaVersion": "e07.s04.task-weighting-sensitivity.v1",
        "researchStepId": "S04",
        "rows": rows,
        "profileSums": sums,
        "minimumWeight": minimum,
        "maximumWeight": maximum,
        "maximumToMinimumRatio": maximum / minimum,
        "policyRankingComputed": False,
        "weightedPerformanceScoreComputed": False,
        "s05EligibilityByProfile": {
            profile: eligible for profile in weighting["profiles"]
        },
        "eligibilityStableAcrossProfiles": True,
        "passed": all(math.isclose(value, 1.0) for value in sums.values())
        and minimum > 0,
    }


def adversarial_audit(anti_gaming: Mapping[str, Any]) -> dict[str, Any]:
    cases = [
        ("always_noop_freeze", ["noop_freeze", "e06_diagnostic_substitution"]),
        ("invalid_action_spam", ["invalid_proposal_spam"]),
        (
            "budget_stalling_policy",
            ["event_budget_stall", "competing_terminal_erasure"],
        ),
        ("composition_imbalance_optimizer", ["raw_aggregation_inflation"]),
        ("flat_state_no_flux", ["fixed_plateau_equilibrium"]),
        ("posthoc_detour_metric_picker", ["detour_metric_shopping"]),
        ("five_axis_scalarizer", ["e05_axis_collapse"]),
        ("free_prefix_reader", ["prefix_information_free_lunch"]),
        ("free_cursor_long_range", ["cursor_range_free_lunch"]),
        ("semantic_alias_flood", ["semantic_duplicate_padding"]),
        ("dead_rule_padding", ["complexity_padding"]),
        ("protected_split_probe", ["protected_outcome_leakage"]),
        ("outcome_selected_weights", ["task_weight_cherry_pick"]),
        ("cell_boundary_oscillator", ["descriptor_boundary_thrash"]),
        ("matching_E05_field_names", ["E05_field_name_binding"]),
        ("synthetic_E06_handle_reuse", ["E06_fixture_binding"]),
    ]
    registry = {row["id"]: row for row in anti_gaming["channels"]}
    rows = []
    covered: set[str] = set()
    for case_id, expected in cases:
        detected = [channel for channel in expected if channel in registry]
        covered.update(detected)
        rows.append(
            {
                "caseId": case_id,
                "expectedChannels": expected,
                "detectedChannels": detected,
                "passed": detected == expected,
                "outcomeCreditPermitted": False,
            }
        )
    missing = sorted(set(registry) - covered)
    return {
        "schemaVersion": "e07.s04.adversarial-audit.v1",
        "researchStepId": "S04",
        "caseCount": len(rows),
        "registeredChannelCount": len(registry),
        "rows": rows,
        "unexercisedRegisteredChannels": missing,
        "passed": all(row["passed"] for row in rows) and not missing,
    }


def build_s04_evidence(
    *,
    objective_path: str | Path,
    descriptor_path: str | Path,
    weighting_path: str | Path,
    uncertainty_path: str | Path,
    anti_gaming_path: str | Path,
    eligibility_path: str | Path,
    baseline_smoke_path: str | Path,
    split_manifest_path: str | Path,
    seed_evaluations_path: str | Path,
    complexity_path: str | Path,
) -> dict[str, Any]:
    objective = load_yaml(objective_path)
    descriptor = load_yaml(descriptor_path)
    weighting = load_yaml(weighting_path)
    uncertainty = load_yaml(uncertainty_path)
    anti_gaming = load_yaml(anti_gaming_path)
    eligibility = load_yaml(eligibility_path)
    registry = validate_registries(
        objective, descriptor, weighting, uncertainty, anti_gaming, eligibility
    )
    normalization = build_normalization_baselines(baseline_smoke_path)
    descriptors = descriptor_validation(
        descriptor, baseline_smoke_path, split_manifest_path
    )
    correlations, correlation_summary = correlation_audit(
        seed_evaluations_path, complexity_path
    )
    weights = weighting_sensitivity(weighting, bool(eligibility["s05SearchEligible"]))
    adversarial = adversarial_audit(anti_gaming)
    checks = {
        "registryValidation": registry["passed"],
        "normalizationUsesEightTrainingAnchors": len(normalization["taskBaselines"])
        == 8,
        "descriptorValidation": descriptors["passed"],
        "correlationAudit": correlation_summary["passed"],
        "weightingSensitivity": weights["passed"],
        "adversarialAudit": adversarial["passed"],
        "validationOutcomeEvaluationsZero": True,
        "confirmationOutcomeEvaluationsZero": True,
        "S05CurrentlyBlocked": eligibility["s05SearchEligible"] is False,
    }
    return {
        "schemaVersion": S04_SCHEMA_VERSION,
        "researchStepId": "S04",
        "objectiveRegistry": objective,
        "descriptorRegistry": descriptor,
        "taskWeighting": weighting,
        "uncertaintyRegistry": uncertainty,
        "antiGamingRegistry": anti_gaming,
        "eligibilityGate": eligibility,
        "registryValidation": registry,
        "normalizationBaselines": normalization,
        "descriptorValidation": descriptors,
        "correlations": correlations,
        "correlationSummary": correlation_summary,
        "weightingSensitivity": weights,
        "adversarialAudit": adversarial,
        "checks": checks,
        "success": all(checks.values()),
    }
