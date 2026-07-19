"""Typed relational grammar parser and deterministic grid scorer for E06 S02.

The grammar is an offline descriptor of observable local relations.  It is not
an executable cell policy and it never receives cell identity, Algotype,
analysis labels, future state, or the S01 global target-membership decision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .targets import Grid, TRANSFORMS, grid_counts, transform_grid


GRAMMAR_CATALOG_VERSION = "e06.s02.grammar-catalog.v1"
CONFLICT_RESOLUTION = "hard_first_then_weighted_soft_v1"
ALLOWED_DIRECTIONS = {"all", "horizontal", "vertical"}
ALLOWED_SIDES = {"any", "top", "bottom", "left", "right"}


class GrammarValidationError(ValueError):
    """Raised when a grammar is malformed or internally contradictory."""


class ConstraintKind(str, Enum):
    EDGE_COUNT = "edge_count"
    NEIGHBOR_COUNT = "neighbor_count"
    MOTIF_COUNT = "motif_count"
    BOUNDARY_COUNT = "boundary_count"
    SYMMETRY = "symmetry"


class Priority(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class GrammarConstraint:
    constraint_id: str
    kind: ConstraintKind
    priority: Priority
    weight: float
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class RelationalGrammar:
    grammar_id: str
    target_id: str
    shape: tuple[int, int]
    vacancy_label: str
    cell_type_counts: Mapping[str, int]
    orientation_transforms: tuple[str, ...]
    conflict_resolution: str
    acceptance_threshold: float
    constraints: tuple[GrammarConstraint, ...]
    audit_only_global_constraints: tuple[str, ...]
    underdetermination_hypothesis: str


def _interval_score(
    observed: float, minimum: float, maximum: float, scale: float
) -> float:
    if minimum <= observed <= maximum:
        return 1.0
    deviation = minimum - observed if observed < minimum else observed - maximum
    return max(0.0, 1.0 - deviation / scale)


def _edge_capacity(shape: tuple[int, int], directions: str) -> int:
    height, width = shape
    horizontal = height * max(0, width - 1)
    vertical = max(0, height - 1) * width
    if directions == "horizontal":
        return horizontal
    if directions == "vertical":
        return vertical
    return horizontal + vertical


def _boundary_sites(
    shape: tuple[int, int], sides: Sequence[str]
) -> set[tuple[int, int]]:
    height, width = shape
    selected: set[tuple[int, int]] = set()
    if "any" in sides:
        sides = ("top", "bottom", "left", "right")
    for side in sides:
        if side == "top":
            selected.update((0, col) for col in range(width))
        elif side == "bottom":
            selected.update((height - 1, col) for col in range(width))
        elif side == "left":
            selected.update((row, 0) for row in range(height))
        elif side == "right":
            selected.update((row, width - 1) for row in range(height))
    return selected


def _offsets(directions: str) -> tuple[tuple[int, int], ...]:
    if directions == "horizontal":
        return ((0, -1), (0, 1))
    if directions == "vertical":
        return ((-1, 0), (1, 0))
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def _edge_count(grid: Grid, tokens: Sequence[str], directions: str) -> int:
    desired = tuple(sorted(tokens))
    height, width = len(grid), len(grid[0])
    count = 0
    if directions in {"all", "horizontal"}:
        for row in range(height):
            for col in range(width - 1):
                if tuple(sorted((grid[row][col], grid[row][col + 1]))) == desired:
                    count += 1
    if directions in {"all", "vertical"}:
        for row in range(height - 1):
            for col in range(width):
                if tuple(sorted((grid[row][col], grid[row + 1][col]))) == desired:
                    count += 1
    return count


def _motif_count(grid: Grid, pattern: Sequence[str]) -> int:
    pattern_height, pattern_width = len(pattern), len(pattern[0])
    height, width = len(grid), len(grid[0])
    count = 0
    for top in range(height - pattern_height + 1):
        for left in range(width - pattern_width + 1):
            matches = True
            for row in range(pattern_height):
                for col in range(pattern_width):
                    token = pattern[row][col]
                    if token != "?" and token != grid[top + row][left + col]:
                        matches = False
                        break
                if not matches:
                    break
            count += int(matches)
    return count


def _foreground_bbox(grid: Grid, vacancy_label: str) -> Grid:
    occupied = [
        (row, col)
        for row in range(len(grid))
        for col in range(len(grid[0]))
        if grid[row][col] != vacancy_label
    ]
    if not occupied:
        return grid
    top = min(row for row, _ in occupied)
    bottom = max(row for row, _ in occupied)
    left = min(col for _, col in occupied)
    right = max(col for _, col in occupied)
    return tuple(
        tuple(grid[row][col] for col in range(left, right + 1))
        for row in range(top, bottom + 1)
    )


def _score_constraint(
    grid: Grid, constraint: GrammarConstraint, vacancy_label: str
) -> dict[str, Any]:
    parameters = constraint.parameters
    minimum = float(parameters.get("min", 0))
    maximum = float(parameters.get("max", minimum))
    scale = float(parameters.get("deviationScale", max(1.0, maximum, minimum)))
    observed: Any
    if constraint.kind == ConstraintKind.EDGE_COUNT:
        observed = _edge_count(grid, parameters["tokens"], parameters["directions"])
        score = _interval_score(observed, minimum, maximum, scale)
        passed = minimum <= observed <= maximum
    elif constraint.kind == ConstraintKind.BOUNDARY_COUNT:
        sites = _boundary_sites((len(grid), len(grid[0])), parameters["sides"])
        observed = sum(grid[row][col] == parameters["token"] for row, col in sites)
        score = _interval_score(observed, minimum, maximum, scale)
        passed = minimum <= observed <= maximum
    elif constraint.kind == ConstraintKind.MOTIF_COUNT:
        observed = _motif_count(grid, parameters["pattern"])
        score = _interval_score(observed, minimum, maximum, scale)
        passed = minimum <= observed <= maximum
    elif constraint.kind == ConstraintKind.SYMMETRY:
        symmetry_grid = (
            grid
            if parameters["frame"] == "domain"
            else _foreground_bbox(grid, vacancy_label)
        )
        transformed = transform_grid(symmetry_grid, parameters["transform"])
        if (len(transformed), len(transformed[0])) != (
            len(symmetry_grid),
            len(symmetry_grid[0]),
        ):
            mismatch_count = len(symmetry_grid) * len(symmetry_grid[0])
        else:
            mismatch_count = sum(
                symmetry_grid[row][col] != transformed[row][col]
                for row in range(len(symmetry_grid))
                for col in range(len(symmetry_grid[0]))
            )
        maximum_mismatches = int(parameters["maxMismatches"])
        observed = mismatch_count
        score = _interval_score(
            mismatch_count,
            0,
            maximum_mismatches,
            float(parameters.get("deviationScale", len(grid) * len(grid[0]))),
        )
        passed = mismatch_count <= maximum_mismatches
    elif constraint.kind == ConstraintKind.NEIGHBOR_COUNT:
        height, width = len(grid), len(grid[0])
        subject = parameters["subject"]
        allowed_neighbors = set(parameters["neighborTokens"])
        satisfying = 0
        subject_count = 0
        neighbor_counts: Counter[int] = Counter()
        for row in range(height):
            for col in range(width):
                if grid[row][col] != subject:
                    continue
                subject_count += 1
                neighbor_count = 0
                for row_offset, col_offset in _offsets(parameters["directions"]):
                    other_row, other_col = row + row_offset, col + col_offset
                    if (
                        0 <= other_row < height
                        and 0 <= other_col < width
                        and grid[other_row][other_col] in allowed_neighbors
                    ):
                        neighbor_count += 1
                neighbor_counts[neighbor_count] += 1
                satisfying += int(minimum <= neighbor_count <= maximum)
        satisfying_fraction = satisfying / subject_count if subject_count else 0.0
        required_fraction = float(parameters["minSubjectFraction"])
        score = (
            min(1.0, satisfying_fraction / required_fraction)
            if required_fraction > 0
            else 1.0
        )
        passed = satisfying_fraction >= required_fraction
        observed = {
            "subjectCount": subject_count,
            "satisfyingCount": satisfying,
            "satisfyingFraction": satisfying_fraction,
            "neighborCountHistogram": {
                str(key): value for key, value in sorted(neighbor_counts.items())
            },
        }
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(f"unhandled constraint kind: {constraint.kind}")
    return {
        "constraintId": constraint.constraint_id,
        "kind": constraint.kind.value,
        "priority": constraint.priority.value,
        "weight": constraint.weight,
        "observed": observed,
        "score": score,
        "passed": passed,
    }


def _score_oriented(
    grid: Grid, grammar: RelationalGrammar, orientation: str
) -> dict[str, Any]:
    oriented = transform_grid(grid, orientation)
    results = [
        _score_constraint(oriented, constraint, grammar.vacancy_label)
        for constraint in grammar.constraints
    ]
    hard_results = [item for item in results if item["priority"] == Priority.HARD.value]
    soft_results = [item for item in results if item["priority"] == Priority.SOFT.value]
    hard_violations = sum(not item["passed"] for item in hard_results)
    soft_weight = sum(item["weight"] for item in soft_results)
    soft_score = (
        sum(item["weight"] * item["score"] for item in soft_results) / soft_weight
        if soft_weight
        else 1.0
    )
    total_weight = sum(item["weight"] for item in results)
    relational_score = (
        sum(item["weight"] * item["score"] for item in results) / total_weight
        if total_weight
        else 1.0
    )
    return {
        "orientation": orientation,
        "hardViolationCount": hard_violations,
        "softScore": soft_score,
        "relationalScore": relational_score,
        "constraints": results,
    }


def score_grid(
    candidate: Sequence[Sequence[str]], grammar: RelationalGrammar
) -> dict[str, Any]:
    """Score a grid under hard-first, best-orientation grammar semantics."""

    grid = tuple(tuple(str(token) for token in row) for row in candidate)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("candidate must be a non-empty rectangular grid")
    if (len(grid), len(grid[0])) != grammar.shape:
        raise ValueError(
            f"candidate shape {(len(grid), len(grid[0]))} != {grammar.shape}"
        )
    observed_counts = grid_counts(grid)
    count_match = observed_counts == Counter(grammar.cell_type_counts)
    orientation_scores = [
        _score_oriented(grid, grammar, orientation)
        for orientation in grammar.orientation_transforms
    ]
    best = max(
        orientation_scores,
        key=lambda item: (
            -item["hardViolationCount"],
            item["softScore"],
            item["relationalScore"],
            item["orientation"],
        ),
    )
    accepted = (
        count_match
        and best["hardViolationCount"] == 0
        and best["softScore"] >= grammar.acceptance_threshold
    )
    return {
        "grammarId": grammar.grammar_id,
        "targetId": grammar.target_id,
        "accepted": accepted,
        "countMatch": count_match,
        "observedCounts": dict(sorted(observed_counts.items())),
        "acceptanceThreshold": grammar.acceptance_threshold,
        **best,
    }


def constraint_to_dict(constraint: GrammarConstraint) -> dict[str, Any]:
    return {
        "constraintId": constraint.constraint_id,
        "kind": constraint.kind.value,
        "priority": constraint.priority.value,
        "weight": constraint.weight,
        "parameters": dict(constraint.parameters),
    }


def grammar_to_dict(grammar: RelationalGrammar) -> dict[str, Any]:
    return {
        "grammarId": grammar.grammar_id,
        "targetId": grammar.target_id,
        "shape": list(grammar.shape),
        "vacancyLabel": grammar.vacancy_label,
        "cellTypeCounts": dict(grammar.cell_type_counts),
        "orientationTransforms": list(grammar.orientation_transforms),
        "conflictResolution": grammar.conflict_resolution,
        "acceptanceThreshold": grammar.acceptance_threshold,
        "constraints": [constraint_to_dict(item) for item in grammar.constraints],
        "auditOnlyGlobalConstraints": list(grammar.audit_only_global_constraints),
        "underdeterminationHypothesis": grammar.underdetermination_hypothesis,
    }


def catalog_to_dict(
    metadata: Mapping[str, Any], grammars: Sequence[RelationalGrammar]
) -> dict[str, Any]:
    return {
        "schemaVersion": metadata["schemaVersion"],
        "researchStepId": metadata["researchStepId"],
        "benchmarkVersion": metadata["benchmarkVersion"],
        "observationContract": metadata["observationContract"],
        "conflictResolution": metadata["conflictResolution"],
        "claimBoundary": metadata["claimBoundary"],
        "grammars": [grammar_to_dict(grammar) for grammar in grammars],
    }


def _parse_constraint(raw: Mapping[str, Any]) -> GrammarConstraint:
    required = {"constraintId", "kind", "priority", "weight", "parameters"}
    if set(raw) != required:
        raise GrammarValidationError(
            f"constraint keys must be exactly {sorted(required)}; got {sorted(raw)}"
        )
    try:
        kind = ConstraintKind(raw["kind"])
        priority = Priority(raw["priority"])
    except ValueError as error:
        raise GrammarValidationError(str(error)) from error
    weight = float(raw["weight"])
    if weight <= 0:
        raise GrammarValidationError("constraint weight must be positive")
    if not isinstance(raw["parameters"], Mapping):
        raise GrammarValidationError("constraint parameters must be a mapping")
    return GrammarConstraint(
        constraint_id=str(raw["constraintId"]),
        kind=kind,
        priority=priority,
        weight=weight,
        parameters=dict(raw["parameters"]),
    )


def parse_grammar(
    raw: Mapping[str, Any], *, reject_contradictions: bool = True
) -> RelationalGrammar:
    required = {
        "grammarId",
        "targetId",
        "shape",
        "vacancyLabel",
        "cellTypeCounts",
        "orientationTransforms",
        "conflictResolution",
        "acceptanceThreshold",
        "constraints",
        "auditOnlyGlobalConstraints",
        "underdeterminationHypothesis",
    }
    if set(raw) != required:
        raise GrammarValidationError(
            f"grammar keys must be exactly {sorted(required)}; got {sorted(raw)}"
        )
    shape = tuple(int(item) for item in raw["shape"])
    if len(shape) != 2 or min(shape) <= 0:
        raise GrammarValidationError("grammar shape must contain two positive integers")
    transforms = tuple(str(item) for item in raw["orientationTransforms"])
    if not transforms or len(transforms) != len(set(transforms)):
        raise GrammarValidationError(
            "orientationTransforms must be non-empty and unique"
        )
    if set(transforms) - TRANSFORMS:
        raise GrammarValidationError(
            "orientationTransforms contains an unknown D4 transform"
        )
    threshold = float(raw["acceptanceThreshold"])
    if not 0 <= threshold <= 1:
        raise GrammarValidationError("acceptanceThreshold must lie in [0, 1]")
    if raw["conflictResolution"] != CONFLICT_RESOLUTION:
        raise GrammarValidationError("unsupported conflictResolution")
    constraints = tuple(_parse_constraint(item) for item in raw["constraints"])
    if not constraints:
        raise GrammarValidationError("grammar must contain at least one constraint")
    constraint_ids = [item.constraint_id for item in constraints]
    if len(constraint_ids) != len(set(constraint_ids)):
        raise GrammarValidationError(
            "constraintId values must be unique within a grammar"
        )
    grammar = RelationalGrammar(
        grammar_id=str(raw["grammarId"]),
        target_id=str(raw["targetId"]),
        shape=(shape[0], shape[1]),
        vacancy_label=str(raw["vacancyLabel"]),
        cell_type_counts={
            str(token): int(count) for token, count in raw["cellTypeCounts"].items()
        },
        orientation_transforms=transforms,
        conflict_resolution=str(raw["conflictResolution"]),
        acceptance_threshold=threshold,
        constraints=constraints,
        audit_only_global_constraints=tuple(
            str(item) for item in raw["auditOnlyGlobalConstraints"]
        ),
        underdetermination_hypothesis=str(raw["underdeterminationHypothesis"]),
    )
    contradictions = detect_contradictions(grammar)
    if reject_contradictions and contradictions:
        raise GrammarValidationError(
            f"grammar {grammar.grammar_id} is contradictory: {'; '.join(contradictions)}"
        )
    return grammar


def _interval(constraint: GrammarConstraint) -> tuple[float, float]:
    return (
        float(constraint.parameters.get("min", 0)),
        float(constraint.parameters.get("max", constraint.parameters.get("min", 0))),
    )


def detect_contradictions(grammar: RelationalGrammar) -> list[str]:
    """Return deterministic structural and cross-constraint contradictions."""

    contradictions: list[str] = []
    height, width = grammar.shape
    cell_count = height * width
    counts = Counter(grammar.cell_type_counts)
    tokens = set(counts)
    if sum(counts.values()) != cell_count:
        contradictions.append(
            f"cellTypeCounts sum {sum(counts.values())} != shape capacity {cell_count}"
        )
    if grammar.vacancy_label not in counts:
        contradictions.append("vacancyLabel is absent from cellTypeCounts")
    interval_groups: dict[tuple[Any, ...], list[tuple[float, float, str]]] = {}
    for constraint in grammar.constraints:
        parameters = constraint.parameters
        minimum, maximum = _interval(constraint)
        if constraint.kind != ConstraintKind.SYMMETRY and minimum > maximum:
            contradictions.append(f"{constraint.constraint_id}: min exceeds max")
        if constraint.kind == ConstraintKind.EDGE_COUNT:
            if set(parameters) != {
                "tokens",
                "directions",
                "min",
                "max",
                "deviationScale",
            }:
                contradictions.append(
                    f"{constraint.constraint_id}: invalid edge_count parameters"
                )
                continue
            edge_tokens = parameters["tokens"]
            if len(edge_tokens) != 2 or set(edge_tokens) - tokens:
                contradictions.append(
                    f"{constraint.constraint_id}: edge tokens are invalid"
                )
            directions = parameters["directions"]
            if directions not in ALLOWED_DIRECTIONS:
                contradictions.append(
                    f"{constraint.constraint_id}: invalid edge directions"
                )
            elif maximum > _edge_capacity(grammar.shape, directions):
                contradictions.append(
                    f"{constraint.constraint_id}: max exceeds available edges"
                )
            if constraint.priority == Priority.HARD:
                signature = ("edge", tuple(sorted(edge_tokens)), directions)
                interval_groups.setdefault(signature, []).append(
                    (minimum, maximum, constraint.constraint_id)
                )
        elif constraint.kind == ConstraintKind.BOUNDARY_COUNT:
            if set(parameters) != {
                "token",
                "sides",
                "min",
                "max",
                "deviationScale",
            }:
                contradictions.append(
                    f"{constraint.constraint_id}: invalid boundary_count parameters"
                )
                continue
            boundary_token = parameters["token"]
            sides = parameters["sides"]
            if boundary_token not in tokens or not sides or set(sides) - ALLOWED_SIDES:
                contradictions.append(
                    f"{constraint.constraint_id}: invalid boundary token or sides"
                )
            else:
                capacity = len(_boundary_sites(grammar.shape, sides))
                if maximum > capacity or minimum > counts[boundary_token]:
                    contradictions.append(
                        f"{constraint.constraint_id}: boundary interval exceeds capacity/count"
                    )
            if constraint.priority == Priority.HARD:
                signature = ("boundary", boundary_token, tuple(sorted(sides)))
                interval_groups.setdefault(signature, []).append(
                    (minimum, maximum, constraint.constraint_id)
                )
        elif constraint.kind == ConstraintKind.MOTIF_COUNT:
            if set(parameters) != {
                "pattern",
                "min",
                "max",
                "deviationScale",
            }:
                contradictions.append(
                    f"{constraint.constraint_id}: invalid motif_count parameters"
                )
                continue
            pattern = parameters["pattern"]
            if (
                not pattern
                or not all(isinstance(row, str) and row for row in pattern)
                or len({len(row) for row in pattern}) != 1
            ):
                contradictions.append(
                    f"{constraint.constraint_id}: motif pattern must be rectangular"
                )
                continue
            motif_tokens = set("".join(pattern)) - {"?"}
            if motif_tokens - tokens:
                contradictions.append(
                    f"{constraint.constraint_id}: motif contains unknown tokens"
                )
            pattern_height, pattern_width = len(pattern), len(pattern[0])
            capacity = max(0, height - pattern_height + 1) * max(
                0, width - pattern_width + 1
            )
            if maximum > capacity:
                contradictions.append(
                    f"{constraint.constraint_id}: motif max exceeds anchor capacity"
                )
            motif_counts = Counter("".join(pattern).replace("?", ""))
            if minimum >= 1 and any(
                motif_counts[token] > counts[token] for token in motif_counts
            ):
                contradictions.append(
                    f"{constraint.constraint_id}: required motif exceeds token counts"
                )
        elif constraint.kind == ConstraintKind.NEIGHBOR_COUNT:
            if set(parameters) != {
                "subject",
                "neighborTokens",
                "directions",
                "min",
                "max",
                "minSubjectFraction",
            }:
                contradictions.append(
                    f"{constraint.constraint_id}: invalid neighbor_count parameters"
                )
                continue
            subject = parameters["subject"]
            neighbor_tokens = parameters["neighborTokens"]
            directions = parameters["directions"]
            capacity = (
                len(_offsets(directions)) if directions in ALLOWED_DIRECTIONS else 0
            )
            if (
                subject not in tokens
                or not neighbor_tokens
                or set(neighbor_tokens) - tokens
                or directions not in ALLOWED_DIRECTIONS
            ):
                contradictions.append(
                    f"{constraint.constraint_id}: invalid neighbor subject/tokens/directions"
                )
            if maximum > capacity:
                contradictions.append(
                    f"{constraint.constraint_id}: max exceeds neighborhood capacity"
                )
            fraction = float(parameters["minSubjectFraction"])
            if not 0 <= fraction <= 1:
                contradictions.append(
                    f"{constraint.constraint_id}: minSubjectFraction outside [0,1]"
                )
        elif constraint.kind == ConstraintKind.SYMMETRY:
            if set(parameters) not in (
                {"transform", "frame", "maxMismatches", "deviationScale"},
                {"transform", "frame", "maxMismatches"},
            ):
                contradictions.append(
                    f"{constraint.constraint_id}: invalid symmetry parameters"
                )
                continue
            transform = parameters["transform"]
            frame = parameters["frame"]
            max_mismatches = int(parameters["maxMismatches"])
            if (
                transform not in TRANSFORMS
                or frame not in {"domain", "foreground_bbox"}
                or not 0 <= max_mismatches <= cell_count
            ):
                contradictions.append(
                    f"{constraint.constraint_id}: invalid symmetry transform/tolerance"
                )
            if max_mismatches == 0 and frame == "domain":
                odd_counts = sum(count % 2 for count in counts.values())
                no_fixed_sites = (
                    (transform == "reflect_vertical" and width % 2 == 0)
                    or (transform == "reflect_horizontal" and height % 2 == 0)
                    or (transform == "rotate180" and height * width % 2 == 0)
                )
                if no_fixed_sites and odd_counts:
                    contradictions.append(
                        f"{constraint.constraint_id}: exact symmetry requires even token counts"
                    )
                if (
                    transform == "rotate180"
                    and height * width % 2 == 1
                    and odd_counts > 1
                ):
                    contradictions.append(
                        f"{constraint.constraint_id}: 180-degree symmetry permits at most one odd token count"
                    )
    for signature, intervals in interval_groups.items():
        if len(intervals) < 2:
            continue
        lower = max(item[0] for item in intervals)
        upper = min(item[1] for item in intervals)
        if lower > upper:
            ids = ",".join(item[2] for item in intervals)
            contradictions.append(
                f"{signature[0]} interval constraints have empty intersection: {ids}"
            )
    return sorted(set(contradictions))


def parse_grammar_catalog(
    raw: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[RelationalGrammar, ...]]:
    required = {
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "observationContract",
        "conflictResolution",
        "claimBoundary",
        "grammars",
    }
    if set(raw) != required:
        raise GrammarValidationError(
            f"catalog keys must be exactly {sorted(required)}; got {sorted(raw)}"
        )
    if (
        raw["schemaVersion"] != GRAMMAR_CATALOG_VERSION
        or raw["researchStepId"] != "S02"
    ):
        raise GrammarValidationError("grammar catalog version/researchStepId mismatch")
    if raw["conflictResolution"] != CONFLICT_RESOLUTION:
        raise GrammarValidationError("catalog conflictResolution mismatch")
    grammars = tuple(parse_grammar(item) for item in raw["grammars"])
    grammar_ids = [grammar.grammar_id for grammar in grammars]
    target_ids = [grammar.target_id for grammar in grammars]
    if len(grammar_ids) != len(set(grammar_ids)) or len(target_ids) != len(
        set(target_ids)
    ):
        raise GrammarValidationError("grammarId and targetId must be unique")
    metadata = {key: raw[key] for key in required - {"grammars"}}
    return metadata, grammars


def load_grammar_catalog(
    path: str | Path,
) -> tuple[Mapping[str, Any], tuple[RelationalGrammar, ...]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise GrammarValidationError("grammar catalog root must be a mapping")
    return parse_grammar_catalog(raw)
