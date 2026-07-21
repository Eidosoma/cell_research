"""Frozen S13 morphology, topology, endpoint-time, and cost diagnostics.

The functions in this module do not define a new completion endpoint.  S02
local grammar fields and S01 global membership fields are deliberately emitted
separately, and the only completion field is their already-frozen conjunction.
"""

from __future__ import annotations

from collections import Counter, deque
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from .grammar import RelationalGrammar, score_grid
from .targets import (
    Grid,
    TargetDefinition,
    evaluate_success,
    exact_equivalence_orbit,
    grid_counts,
    transform_grid,
    vacancy_hole_count,
)


MORPHOMETRIC_CATALOG_VERSION = "e06.s13.morphometric-catalog.v1"
TRANSFORM_ORDER = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "reflect_horizontal",
    "reflect_vertical",
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
)

# Every available ledger component stays separate.  A missing source field is
# represented as null, never as a structural zero.
COST_SOURCE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "S09": (
        "acceptedMovements",
        "submittedProposals",
        "conflictLosses",
        "invalidProposals",
        "totalGraphDisplacement",
        "observationCommunicatedBitsUpperBound",
        "observationUtilityEvaluations",
        "channelConfigurationBits",
        "channelTotalInformationBits",
    ),
    "S10": (
        "externalDisplacedEntities",
        "externalDisplacedCells",
        "externalDisplacedVacancies",
        "externalGraphDisplacement",
        "gateAttemptedProposals",
        "suppressedProposals",
        "suppressedRouteSiteClaims",
        "activeGateTransitions",
        "acceptedMovements",
        "submittedProposals",
        "conflictLosses",
        "invalidProposals",
        "totalGraphDisplacement",
        "observationCommunicatedBitsUpperBound",
        "observationUtilityEvaluations",
        "channelConfigurationBits",
        "channelTotalInformationBits",
        "totalInterventionCostUnits",
    ),
    "S11": (
        "group0SubmittedProposals",
        "group1SubmittedProposals",
        "group0AcceptedActorProposals",
        "group1AcceptedActorProposals",
        "group0ConflictLosses",
        "group1ConflictLosses",
        "crossGroupActorTargetProposals",
        "group0ActivationOpportunities",
        "group1ActivationOpportunities",
        "acceptedMovements",
        "submittedProposals",
        "conflictLosses",
        "invalidProposals",
        "totalGraphDisplacement",
        "observationCommunicatedBitsUpperBound",
        "observationUtilityEvaluations",
        "channelConfigurationBits",
        "channelTotalInformationBits",
        "chimeraConfigurationBits",
        "interventionControllerInputBits",
        "interventionConfigurationBits",
        "interventionAssignmentChanges",
        "interventionPolicySwitches",
        "interventionGrammarSwitches",
        "interventionPhysicalDisplacement",
    ),
    "S12": (
        "externalDisplacedEntities",
        "externalDisplacedCells",
        "externalDisplacedVacancies",
        "externalGraphDisplacement",
        "acceptedMovements",
        "submittedProposals",
        "conflictLosses",
        "invalidProposals",
        "totalGraphDisplacement",
        "observationCommunicatedBitsUpperBound",
        "policyLogicalSourceReads",
        "policyUtilityEvaluations",
        "policyComparisonOperations",
        "configurationBits",
        "controllerInputBits",
        "policyDeliveryBits",
        "addressBits",
        "channelTotalInformationBits",
        "totalInformationBitsIncludingObservation",
        "sourceScalarReads",
        "controllerStateReads",
        "controllerComputeUnits",
        "actuationAttempts",
        "actuationSuccesses",
        "overrideActionUnits",
        "directMovementGraphDisplacement",
        "suppressedNativeActions",
        "channelOpportunityCostUnits",
        "allocatedNativeActorSlots",
        "usedNativeActorSlots",
        "directRecipientQuerySlots",
        "foregoneNativeActorSlots",
        "externalInterventionGraphDisplacement",
        "totalSourceWorkUnits",
        "totalComputationUnits",
        "totalOpportunityCostUnits",
    ),
}
ALL_COST_SOURCE_COLUMNS = tuple(
    sorted({column for columns in COST_SOURCE_COLUMNS.values() for column in columns})
)


def coerce_grid(candidate: Sequence[Sequence[str]]) -> Grid:
    grid = tuple(tuple(str(token) for token in row) for row in candidate)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("candidate must be a non-empty rectangular grid")
    return grid


def grid_from_reference_key(reference_key: str) -> Grid:
    rows = reference_key.split("\n")
    return coerce_grid(rows)


def neighbors(row: int, col: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other_row, other_col = row + row_delta, col + col_delta
        if 0 <= other_row < height and 0 <= other_col < width:
            yield other_row, other_col


def token_components(grid: Sequence[Sequence[str]], token: str) -> tuple[frozenset[tuple[int, int]], ...]:
    frozen = coerce_grid(grid)
    height, width = len(frozen), len(frozen[0])
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if frozen[row][col] == token
    }
    output: list[frozenset[tuple[int, int]]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component = {seed}
        queue = deque([seed])
        while queue:
            row, col = queue.popleft()
            for other in neighbors(row, col, height, width):
                if other in unseen:
                    unseen.remove(other)
                    component.add(other)
                    queue.append(other)
        output.append(frozenset(component))
    return tuple(output)


def component_counts(grid: Sequence[Sequence[str]], tokens: Iterable[str] | None = None) -> dict[str, int]:
    frozen = coerce_grid(grid)
    selected = sorted(set(tokens) if tokens is not None else set(grid_counts(frozen)))
    return {token: len(token_components(frozen, token)) for token in selected}


def heterotypic_boundary_edges(grid: Sequence[Sequence[str]]) -> int:
    frozen = coerce_grid(grid)
    height, width = len(frozen), len(frozen[0])
    horizontal = sum(
        frozen[row][col] != frozen[row][col + 1]
        for row in range(height)
        for col in range(width - 1)
    )
    vertical = sum(
        frozen[row][col] != frozen[row + 1][col]
        for row in range(height - 1)
        for col in range(width)
    )
    return int(horizontal + vertical)


def foreground_bbox(grid: Sequence[Sequence[str]], vacancy_label: str) -> Grid:
    frozen = coerce_grid(grid)
    occupied = [
        (row, col)
        for row in range(len(frozen))
        for col in range(len(frozen[0]))
        if frozen[row][col] != vacancy_label
    ]
    if not occupied:
        return frozen
    top = min(row for row, _ in occupied)
    bottom = max(row for row, _ in occupied)
    left = min(col for _, col in occupied)
    right = max(col for _, col in occupied)
    return tuple(
        tuple(frozen[row][col] for col in range(left, right + 1))
        for row in range(top, bottom + 1)
    )


def exact_stabilizers(grid: Sequence[Sequence[str]]) -> tuple[str, ...]:
    frozen = coerce_grid(grid)
    stabilizers = []
    for transform in TRANSFORM_ORDER:
        transformed = transform_grid(frozen, transform)
        if (
            len(transformed) == len(frozen)
            and len(transformed[0]) == len(frozen[0])
            and transformed == frozen
        ):
            stabilizers.append(transform)
    return tuple(stabilizers)


def symmetry_mismatch_fraction(
    candidate: Sequence[Sequence[str]],
    reference: Sequence[Sequence[str]],
    *,
    vacancy_label: str,
    translation_mode: str,
) -> tuple[float | None, int]:
    candidate_grid = coerce_grid(candidate)
    reference_grid = coerce_grid(reference)
    if translation_mode == "bounded_foreground":
        candidate_frame = foreground_bbox(candidate_grid, vacancy_label)
        reference_frame = foreground_bbox(reference_grid, vacancy_label)
    elif translation_mode == "none":
        candidate_frame = candidate_grid
        reference_frame = reference_grid
    else:
        raise ValueError(f"unsupported translation mode: {translation_mode}")
    stabilizers = tuple(
        transform for transform in exact_stabilizers(reference_frame) if transform != "identity"
    )
    if not stabilizers:
        return None, 0
    site_count = len(reference_frame) * len(reference_frame[0])
    mismatches = []
    for transform in stabilizers:
        transformed = transform_grid(candidate_frame, transform)
        if (
            len(candidate_frame) != len(reference_frame)
            or len(candidate_frame[0]) != len(reference_frame[0])
            or len(transformed) != len(candidate_frame)
            or len(transformed[0]) != len(candidate_frame[0])
        ):
            mismatches.append(1.0)
            continue
        mismatches.append(
            sum(
                candidate_frame[row][col] != transformed[row][col]
                for row in range(len(candidate_frame))
                for col in range(len(candidate_frame[0]))
            )
            / site_count
        )
    return float(np.mean(mismatches)), len(stabilizers)


def occupied_region_purity(
    candidate: Sequence[Sequence[str]],
    reference: Sequence[Sequence[str]],
    vacancy_label: str,
) -> float | None:
    candidate_grid = coerce_grid(candidate)
    reference_grid = coerce_grid(reference)
    if (len(candidate_grid), len(candidate_grid[0])) != (
        len(reference_grid),
        len(reference_grid[0]),
    ):
        raise ValueError("candidate and reference shapes differ")
    regions = []
    for token in sorted(set(grid_counts(reference_grid)) - {vacancy_label}):
        regions.extend(token_components(reference_grid, token))
    total = sum(len(region) for region in regions)
    if total == 0:
        return None
    modal_sum = 0
    for region in regions:
        counts = Counter(candidate_grid[row][col] for row, col in region)
        modal_sum += max(counts.values())
    return modal_sum / total


def categorical_transport_distance(
    candidate: Sequence[Sequence[str]], reference: Sequence[Sequence[str]]
) -> tuple[float | None, int | None, bool]:
    """Exact categorical Manhattan 1-Wasserstein distance for equal counts."""

    candidate_grid = coerce_grid(candidate)
    reference_grid = coerce_grid(reference)
    if (len(candidate_grid), len(candidate_grid[0])) != (
        len(reference_grid),
        len(reference_grid[0]),
    ):
        raise ValueError("candidate and reference shapes differ")
    if grid_counts(candidate_grid) != grid_counts(reference_grid):
        return None, None, False
    height, width = len(candidate_grid), len(candidate_grid[0])
    raw_cost = 0
    for token in sorted(grid_counts(reference_grid)):
        candidate_positions = np.asarray(
            [
                (row, col)
                for row in range(height)
                for col in range(width)
                if candidate_grid[row][col] == token
            ],
            dtype=np.int16,
        )
        reference_positions = np.asarray(
            [
                (row, col)
                for row in range(height)
                for col in range(width)
                if reference_grid[row][col] == token
            ],
            dtype=np.int16,
        )
        if len(candidate_positions) == 0:
            continue
        costs = np.abs(
            candidate_positions[:, None, :] - reference_positions[None, :, :]
        ).sum(axis=2)
        rows, columns = linear_sum_assignment(costs)
        raw_cost += int(costs[rows, columns].sum())
    site_count = height * width
    diameter = max(1, height + width - 2)
    return raw_cost / (site_count * diameter), raw_cost, True


def _interval_error(observed: int, bounds: Sequence[int]) -> int:
    lower, upper = int(bounds[0]), int(bounds[1])
    if observed < lower:
        return lower - observed
    if observed > upper:
        return observed - upper
    return 0


def score_morphology(
    candidate: Sequence[Sequence[str]],
    target: TargetDefinition,
    grammar: RelationalGrammar,
    *,
    equivalence_orbit: Sequence[Grid] | None = None,
) -> dict[str, Any]:
    """Apply the complete frozen S13 spatial panel to one canonical grid."""

    grid = coerce_grid(candidate)
    orbit = tuple(equivalence_orbit or exact_equivalence_orbit(target))
    global_audit = evaluate_success(grid, target, equivalence_orbit=orbit)
    local_audit = score_grid(grid, grammar)
    reference = grid_from_reference_key(global_audit["referenceKey"])
    transport, transport_raw, transport_valid = categorical_transport_distance(
        grid, reference
    )
    counts = component_counts(grid, target.cell_type_counts)
    component_excess = sum(
        _interval_error(counts[token], bounds)
        for token, bounds in target.success["componentCounts"].items()
    )
    holes = vacancy_hole_count(grid, target.vacancy_label)
    hole_error = _interval_error(holes, target.success["vacancyHoles"])
    boundary_edges = heterotypic_boundary_edges(grid)
    reference_edges = heterotypic_boundary_edges(reference)
    symmetry_error, symmetry_count = symmetry_mismatch_fraction(
        grid,
        reference,
        vacancy_label=target.vacancy_label,
        translation_mode=str(target.equivalence["translation"]),
    )
    site_count = len(grid) * len(grid[0])
    return {
        "targetId": target.target_id,
        "grammarId": grammar.grammar_id,
        "s02GrammarAccepted": bool(local_audit["accepted"]),
        "s02SoftScore": float(local_audit["softScore"]),
        "s02RelationalScore": float(local_audit["relationalScore"]),
        "s02HardViolationCount": int(local_audit["hardViolationCount"]),
        "s01GlobalSuccess": bool(global_audit["success"]),
        "s01CountMatch": bool(global_audit["countMatch"]),
        "s01GeometryMatch": bool(global_audit["geometryMatch"]),
        "s01ComponentMatch": bool(global_audit["componentMatch"]),
        "s01TopologyMatch": bool(global_audit["topologyMatch"]),
        "conjunctiveCompletion": bool(
            local_audit["accepted"] and global_audit["success"]
        ),
        "s01ReferenceKey": str(global_audit["referenceKey"]),
        "s01EquivalenceOrbitSize": int(global_audit["equivalenceOrbitSize"]),
        "s01MismatchCount": int(global_audit["mismatchCount"]),
        "imageDistanceModuloEquivalence": float(global_audit["mismatchFraction"]),
        "s01BoundaryViolationCount": int(global_audit["boundaryViolationCount"]),
        "transportDistance": transport,
        "transportRawCost": transport_raw,
        "transportCompositionValid": transport_valid,
        "componentCountsJson": __import__("json").dumps(
            counts, sort_keys=True, separators=(",", ":")
        ),
        "componentExcess": int(component_excess),
        "heterotypicBoundaryEdges": int(boundary_edges),
        "boundaryLengthNormalized": boundary_edges / sqrt(site_count),
        "targetBoundaryRelativeError": abs(boundary_edges - reference_edges)
        / max(1, reference_edges),
        "occupiedRegionPurity": occupied_region_purity(
            grid, reference, target.vacancy_label
        ),
        "symmetryMismatchFraction": symmetry_error,
        "symmetryTransformCount": int(symmetry_count),
        "vacancyHoles": int(holes),
        "holeCountError": int(hole_error),
        "boundedSquareCalibrationOnly": True,
    }


def block_upscale(grid: Sequence[Sequence[str]], scale: int) -> Grid:
    if scale < 1:
        raise ValueError("scale must be positive")
    frozen = coerce_grid(grid)
    rows = []
    for row in frozen:
        expanded = tuple(token for token in row for _ in range(scale))
        rows.extend(expanded for _ in range(scale))
    return tuple(rows)


def standardize_endpoint_time(step_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Map frozen step-specific event fields without inventing maintenance time."""

    endpoint_type = "not_applicable"
    event = False
    time_value: float | None = None
    eligible = False
    if step_id == "S09":
        endpoint_type = "formation"
        eligible = True
        event = bool(row["conjunctiveCompletionByBudget"])
        time_value = row.get("firstCompletionTransition") if event else None
    elif step_id == "S10" and bool(row.get("repairEligible", False)):
        endpoint_type = "repair"
        eligible = True
        event = bool(row["lesionRepairByBudget"])
        time_value = row.get("firstRepairTransition") if event else None
    elif step_id == "S11" and not bool(row.get("initialConjunctiveCompletion", False)):
        endpoint_type = "formation"
        eligible = True
        event = bool(row["conjunctiveCompletionByBudget"])
        time_value = row.get("firstCompletionTransition") if event else None
    elif step_id == "S12":
        if bool(row.get("formationEligible", False)):
            endpoint_type = "formation"
            eligible = True
        elif bool(row.get("repairEligible", False)):
            endpoint_type = "repair"
            eligible = True
        if eligible:
            event = bool(row["conjunctiveCompletionByBudget"])
            time_value = row.get("firstCompletionTransition") if event else None
    if time_value is not None and bool(np.isnan(float(time_value))):
        time_value = None
    censored = bool(eligible and not event)
    analysis_time = float(time_value) if event and time_value is not None else (
        float(row["eventBudgetTransitions"]) if eligible else None
    )
    return {
        "timeEndpointType": endpoint_type,
        "timeEligible": eligible,
        "timeEventObserved": event,
        "timeToEndpoint": float(time_value) if time_value is not None else None,
        "timeRightCensored": censored,
        "timeAnalysisTransition": analysis_time,
    }


def standardize_cost_components(step_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if step_id not in COST_SOURCE_COLUMNS:
        raise ValueError(f"unsupported source step: {step_id}")
    applicable = set(COST_SOURCE_COLUMNS[step_id])
    output = {}
    for column in ALL_COST_SOURCE_COLUMNS:
        output[f"cost_{column}"] = row.get(column) if column in applicable else None
    return output


def load_morphometric_catalog(path: str | Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("morphometric catalog root must be a mapping")
    validate_morphometric_catalog(raw)
    return raw


def validate_morphometric_catalog(raw: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion",
        "researchStepId",
        "title",
        "frozenQuestion",
        "claimBoundary",
        "canonicalContracts",
        "equivalenceHandling",
        "metricDefinitions",
        "calibrationFixtures",
        "resolutionRobustness",
        "disagreementRules",
        "decisionRules",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"morphometric catalog missing keys: {missing}")
    if raw["schemaVersion"] != MORPHOMETRIC_CATALOG_VERSION:
        raise ValueError("morphometric catalog version mismatch")
    if raw["researchStepId"] != "S13":
        raise ValueError("morphometric catalog researchStepId mismatch")
    expected_metrics = {
        "grammarSatisfaction",
        "imageDistanceModuloEquivalence",
        "categoricalEarthMoversDistance",
        "connectedComponents",
        "boundaryLength",
        "regionPurity",
        "symmetry",
        "holes",
        "endpointTime",
        "componentWiseCost",
    }
    if set(raw["metricDefinitions"]) != expected_metrics:
        raise ValueError("morphometric catalog metric panel mismatch")
    if raw["decisionRules"]["completionRedefinition"] != "forbidden":
        raise ValueError("S13 may not redefine completion")
    if raw["decisionRules"]["priorOutcomeClassificationChanges"] != "forbidden":
        raise ValueError("S13 may not alter prior classifications")
    if raw["metricDefinitions"]["componentWiseCost"]["scalarCollapse"] != "forbidden":
        raise ValueError("S13 may not collapse component-wise costs")
