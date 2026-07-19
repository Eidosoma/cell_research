"""Executable target sets for the E06 two-dimensional benchmark.

S01 defines geometry, equivalence, feasibility, and success membership only.
The ``localGrammarCandidate`` records are deliberately inert metadata for S02;
this module does not parse them into a policy or score.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


CATALOG_VERSION = "e06.s01.target-catalog.v1"
REQUIRED_FAMILIES = {
    "stripes",
    "layers",
    "rings",
    "bilateral",
    "separated_regions",
    "holes",
}
TRANSFORMS = {
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "reflect_horizontal",
    "reflect_vertical",
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
}

Grid = tuple[tuple[str, ...], ...]
Coordinate = tuple[int, int]
Swap = tuple[Coordinate, Coordinate]


@dataclass(frozen=True)
class TargetDefinition:
    """Validated target definition and its exact small fixture."""

    target_id: str
    family: str
    title: str
    description: str
    vacancy_label: str
    grid: Grid
    cell_type_counts: Mapping[str, int]
    equivalence: Mapping[str, Any]
    success: Mapping[str, Any]
    local_grammar_candidate: Mapping[str, Any]
    validation: Mapping[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.grid), len(self.grid[0])


def _grid_from_rows(rows: Sequence[str]) -> Grid:
    if not rows or not all(isinstance(row, str) and row for row in rows):
        raise ValueError("fixture rows must be non-empty strings")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("fixture rows must have equal width")
    return tuple(tuple(row) for row in rows)


def _grid_key(grid: Grid) -> str:
    return "\n".join("".join(row) for row in grid)


def grid_counts(grid: Grid) -> Counter[str]:
    return Counter(token for row in grid for token in row)


def _transform(grid: Grid, name: str) -> Grid:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform: {name}")
    if name == "identity":
        return grid
    if name == "rotate90":
        return tuple(tuple(row) for row in zip(*grid[::-1]))
    if name == "rotate180":
        return tuple(tuple(reversed(row)) for row in reversed(grid))
    if name == "rotate270":
        return tuple(tuple(row) for row in zip(*grid))[::-1]
    if name == "reflect_horizontal":
        return tuple(reversed(grid))
    if name == "reflect_vertical":
        return tuple(tuple(reversed(row)) for row in grid)
    if name == "reflect_main_diagonal":
        return tuple(tuple(row) for row in zip(*grid))
    # Anti-diagonal reflection is a 180-degree rotation after main-diagonal reflection.
    transposed = tuple(tuple(row) for row in zip(*grid))
    return tuple(tuple(reversed(row)) for row in reversed(transposed))


def _bounded_foreground_translations(grid: Grid, vacancy_label: str) -> set[Grid]:
    height, width = len(grid), len(grid[0])
    occupied = [
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] != vacancy_label
    ]
    if not occupied:
        return {grid}
    top = min(row for row, _ in occupied)
    bottom = max(row for row, _ in occupied)
    left = min(col for _, col in occupied)
    right = max(col for _, col in occupied)
    motif = tuple(
        tuple(grid[row][col] for col in range(left, right + 1))
        for row in range(top, bottom + 1)
    )
    motif_height, motif_width = len(motif), len(motif[0])
    translated: set[Grid] = set()
    for target_top in range(height - motif_height + 1):
        for target_left in range(width - motif_width + 1):
            canvas = [[vacancy_label for _ in range(width)] for _ in range(height)]
            for row in range(motif_height):
                for col in range(motif_width):
                    canvas[target_top + row][target_left + col] = motif[row][col]
            translated.add(tuple(tuple(row) for row in canvas))
    return translated


def exact_equivalence_orbit(target: TargetDefinition) -> tuple[Grid, ...]:
    """Enumerate the finite exact orbit declared by a target."""

    orbit: set[Grid] = set()
    translation = target.equivalence["translation"]
    for transform_name in target.equivalence["transforms"]:
        transformed = _transform(target.grid, transform_name)
        if (len(transformed), len(transformed[0])) != target.shape:
            raise ValueError(
                f"{target.target_id}: transform {transform_name} changes fixture shape"
            )
        if translation == "none":
            orbit.add(transformed)
        elif translation == "bounded_foreground":
            orbit.update(
                _bounded_foreground_translations(transformed, target.vacancy_label)
            )
        else:
            raise ValueError(
                f"{target.target_id}: unsupported translation mode {translation}"
            )
    return tuple(sorted(orbit, key=_grid_key))


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[Coordinate]:
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other_row, other_col = row + delta_row, col + delta_col
        if 0 <= other_row < height and 0 <= other_col < width:
            yield other_row, other_col


def _component_count(grid: Grid, token: str) -> int:
    height, width = len(grid), len(grid[0])
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] == token
    }
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            row, col = queue.popleft()
            for neighbor in _neighbors(row, col, height, width):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return components


def vacancy_hole_count(grid: Grid, vacancy_label: str) -> int:
    """Count 4-connected vacancy components that do not touch the grid boundary."""

    height, width = len(grid), len(grid[0])
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] == vacancy_label
    }
    holes = 0
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        touches_boundary = seed[0] in (0, height - 1) or seed[1] in (0, width - 1)
        while queue:
            row, col = queue.popleft()
            for neighbor in _neighbors(row, col, height, width):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
                    touches_boundary = (
                        touches_boundary
                        or neighbor[0]
                        in (
                            0,
                            height - 1,
                        )
                        or neighbor[1] in (0, width - 1)
                    )
        if not touches_boundary:
            holes += 1
    return holes


def _interface_mask(grid: Grid, radius: int) -> set[Coordinate]:
    height, width = len(grid), len(grid[0])
    interface = set()
    for row in range(height):
        for col in range(width):
            if any(
                grid[other_row][other_col] != grid[row][col]
                for other_row, other_col in _neighbors(row, col, height, width)
            ):
                interface.add((row, col))
    expanded = set(interface)
    for row in range(height):
        for col in range(width):
            if any(
                abs(row - interface_row) + abs(col - interface_col) <= radius
                for interface_row, interface_col in interface
            ):
                expanded.add((row, col))
    return expanded


def _coerce_candidate(
    candidate: Sequence[Sequence[str]], shape: tuple[int, int]
) -> Grid:
    grid = tuple(tuple(str(token) for token in row) for row in candidate)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("candidate must be a non-empty rectangular grid")
    if (len(grid), len(grid[0])) != shape:
        raise ValueError(f"candidate shape {len(grid), len(grid[0])} != {shape}")
    return grid


def evaluate_success(
    candidate: Sequence[Sequence[str]], target: TargetDefinition
) -> dict[str, Any]:
    """Evaluate count, equivalence-aware mismatch, component, and topology gates."""

    grid = _coerce_candidate(candidate, target.shape)
    observed_counts = grid_counts(grid)
    count_match = observed_counts == Counter(target.cell_type_counts)
    allowed_tokens = set(target.cell_type_counts)
    unknown_tokens = sorted(set(observed_counts) - allowed_tokens)

    tolerance = target.success
    best: dict[str, Any] | None = None
    for reference in exact_equivalence_orbit(target):
        mismatches = {
            (row, col)
            for row in range(target.shape[0])
            for col in range(target.shape[1])
            if grid[row][col] != reference[row][col]
        }
        mismatch_count = len(mismatches)
        mismatch_fraction = mismatch_count / (target.shape[0] * target.shape[1])
        if tolerance["mismatchRegion"] == "strict":
            boundary_violations = mismatch_count
        else:
            allowed_mask = _interface_mask(reference, tolerance["interfaceBandRadius"])
            boundary_violations = len(mismatches - allowed_mask)
        record = {
            "referenceKey": _grid_key(reference),
            "mismatchCount": mismatch_count,
            "mismatchFraction": mismatch_fraction,
            "boundaryViolationCount": boundary_violations,
        }
        rank = (
            boundary_violations,
            mismatch_count,
            mismatch_fraction,
            record["referenceKey"],
        )
        if best is None or rank < best["_rank"]:
            best = {**record, "_rank": rank}
    assert best is not None

    component_counts = {
        token: _component_count(grid, token)
        for token in target.success["componentCounts"]
    }
    component_match = all(
        bounds[0] <= component_counts[token] <= bounds[1]
        for token, bounds in target.success["componentCounts"].items()
    )
    holes = vacancy_hole_count(grid, target.vacancy_label)
    hole_bounds = target.success["vacancyHoles"]
    topology_match = hole_bounds[0] <= holes <= hole_bounds[1]
    geometry_match = (
        best["mismatchCount"] <= tolerance["maxMismatches"]
        and best["mismatchFraction"] <= tolerance["maxMismatchFraction"]
        and best["boundaryViolationCount"] == 0
    )
    success = (
        count_match
        and not unknown_tokens
        and geometry_match
        and component_match
        and topology_match
    )
    best.pop("_rank")
    return {
        "success": success,
        "targetId": target.target_id,
        "countMatch": count_match,
        "unknownTokens": unknown_tokens,
        "geometryMatch": geometry_match,
        "componentMatch": component_match,
        "topologyMatch": topology_match,
        "componentCounts": component_counts,
        "vacancyHoles": holes,
        "equivalenceOrbitSize": len(exact_equivalence_orbit(target)),
        **best,
    }


def _snake_coordinates(height: int, width: int) -> tuple[Coordinate, ...]:
    return tuple(
        (row, col)
        for row in range(height)
        for col in (range(width) if row % 2 == 0 else range(width - 1, -1, -1))
    )


def deterministic_scramble(grid: Grid) -> Grid:
    """Create a deterministic, count-preserving non-target permutation."""

    height, width = len(grid), len(grid[0])
    path = _snake_coordinates(height, width)
    values = [grid[row][col] for row, col in path]
    for shift in range(1, len(values)):
        shifted = values[shift:] + values[:shift]
        if shifted != values:
            canvas = [list(row) for row in grid]
            for (row, col), value in zip(path, shifted):
                canvas[row][col] = value
            return tuple(tuple(row) for row in canvas)
    return grid


def adjacent_swap_witness(start: Grid, goal: Grid) -> tuple[Swap, ...]:
    """Construct a legal 4-neighbor adjacent-swap path from start to goal.

    The grid's serpentine Hamiltonian path turns the problem into stable adjacent
    swaps on a line. Duplicate labels and vacancies are supported.
    """

    if (len(start), len(start[0])) != (len(goal), len(goal[0])):
        raise ValueError("start and goal shapes differ")
    if grid_counts(start) != grid_counts(goal):
        raise ValueError("start and goal counts differ")
    path = _snake_coordinates(len(goal), len(goal[0]))
    current = [start[row][col] for row, col in path]
    desired = [goal[row][col] for row, col in path]
    swaps: list[Swap] = []
    for destination in range(len(current)):
        try:
            source = current.index(desired[destination], destination)
        except ValueError as error:  # Defensive: counts were already checked.
            raise ValueError("start cannot be matched to goal") from error
        while source > destination:
            left, right = source - 1, source
            current[left], current[right] = current[right], current[left]
            swaps.append((path[left], path[right]))
            source -= 1
    if current != desired:
        raise AssertionError("witness construction did not reach the goal")
    return tuple(swaps)


def apply_swap_witness(start: Grid, swaps: Sequence[Swap]) -> Grid:
    grid = [list(row) for row in start]
    height, width = len(grid), len(grid[0])
    for first, second in swaps:
        if (
            first == second
            or abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
        ):
            raise ValueError(f"non-adjacent swap in witness: {first}, {second}")
        if not all(
            0 <= row < height and 0 <= col < width for row, col in (first, second)
        ):
            raise ValueError(f"out-of-bounds swap in witness: {first}, {second}")
        grid[first[0]][first[1]], grid[second[0]][second[1]] = (
            grid[second[0]][second[1]],
            grid[first[0]][first[1]],
        )
    return tuple(tuple(row) for row in grid)


def _parse_target(raw: Mapping[str, Any], vacancy_label: str) -> TargetDefinition:
    counts = {
        token: int(metadata["count"]) for token, metadata in raw["cellTypes"].items()
    }
    return TargetDefinition(
        target_id=str(raw["targetId"]),
        family=str(raw["family"]),
        title=str(raw["title"]),
        description=str(raw["description"]),
        vacancy_label=vacancy_label,
        grid=_grid_from_rows(raw["fixture"]["rows"]),
        cell_type_counts=counts,
        equivalence=raw["equivalence"],
        success=raw["success"],
        local_grammar_candidate=raw["localGrammarCandidate"],
        validation=raw["validation"],
    )


def load_target_catalog(
    path: str | Path,
) -> tuple[Mapping[str, Any], tuple[TargetDefinition, ...]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("catalog root must be a mapping")
    validate_target_catalog(raw)
    vacancy_label = str(raw["vacancyLabel"])
    return raw, tuple(_parse_target(item, vacancy_label) for item in raw["targets"])


def validate_target_catalog(raw: Mapping[str, Any]) -> None:
    required_root = {
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "coordinateConvention",
        "vacancyLabel",
        "movementFeasibilityAssumption",
        "claimBoundary",
        "targets",
    }
    missing_root = sorted(required_root - set(raw))
    if missing_root:
        raise ValueError(f"catalog missing root keys: {missing_root}")
    if raw["schemaVersion"] != CATALOG_VERSION or raw["researchStepId"] != "S01":
        raise ValueError("catalog schemaVersion/researchStepId mismatch")
    if not isinstance(raw["targets"], list) or not raw["targets"]:
        raise ValueError("catalog targets must be a non-empty list")
    vacancy_label = str(raw["vacancyLabel"])
    target_ids: set[str] = set()
    families: set[str] = set()
    for item in raw["targets"]:
        required = {
            "targetId",
            "family",
            "title",
            "description",
            "fixture",
            "cellTypes",
            "equivalence",
            "success",
            "localGrammarCandidate",
            "validation",
        }
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"target missing keys: {missing}")
        target_id = str(item["targetId"])
        if target_id in target_ids:
            raise ValueError(f"duplicate targetId: {target_id}")
        target_ids.add(target_id)
        families.add(str(item["family"]))
        grid = _grid_from_rows(item["fixture"]["rows"])
        declared_shape = tuple(item["fixture"]["shape"])
        if declared_shape != (len(grid), len(grid[0])):
            raise ValueError(f"{target_id}: declared shape does not match rows")
        declared_counts = {
            str(token): int(metadata["count"])
            for token, metadata in item["cellTypes"].items()
        }
        if vacancy_label not in declared_counts:
            raise ValueError(f"{target_id}: vacancy label missing from cellTypes")
        if grid_counts(grid) != Counter(declared_counts):
            raise ValueError(f"{target_id}: fixture counts do not match cellTypes")
        transforms = item["equivalence"].get("transforms", [])
        if not transforms or len(transforms) != len(set(transforms)):
            raise ValueError(f"{target_id}: transforms must be non-empty and unique")
        if set(transforms) - TRANSFORMS:
            raise ValueError(f"{target_id}: unknown transform")
        if item["equivalence"].get("translation") not in {
            "none",
            "bounded_foreground",
        }:
            raise ValueError(f"{target_id}: invalid translation")
        success = item["success"]
        required_success = {
            "countMode",
            "maxMismatches",
            "maxMismatchFraction",
            "mismatchRegion",
            "interfaceBandRadius",
            "componentCounts",
            "vacancyHoles",
            "requiredMetrics",
        }
        if required_success - set(success):
            raise ValueError(f"{target_id}: incomplete success specification")
        if success["countMode"] != "exact":
            raise ValueError(f"{target_id}: only exact countMode is supported in S01")
        if success["mismatchRegion"] not in {"strict", "interface_band"}:
            raise ValueError(f"{target_id}: invalid mismatchRegion")
        if not 0 <= int(success["maxMismatches"]) <= len(grid) * len(grid[0]):
            raise ValueError(f"{target_id}: invalid maxMismatches")
        if not 0 <= float(success["maxMismatchFraction"]) <= 1:
            raise ValueError(f"{target_id}: invalid maxMismatchFraction")
        for token, bounds in success["componentCounts"].items():
            if (
                token not in declared_counts
                or len(bounds) != 2
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{target_id}: invalid component bounds for {token}")
        if (
            len(success["vacancyHoles"]) != 2
            or success["vacancyHoles"][0] > success["vacancyHoles"][1]
        ):
            raise ValueError(f"{target_id}: invalid vacancyHoles bounds")
        grammar = item["localGrammarCandidate"]
        for key in (
            "candidateId",
            "orientationOrBoundaryCue",
            "desiredNeighbors",
            "forbiddenNeighbors",
            "motifs",
            "knownUnderdetermination",
        ):
            if key not in grammar:
                raise ValueError(f"{target_id}: grammar candidate missing {key}")
    missing_families = REQUIRED_FAMILIES - families
    if missing_families:
        raise ValueError(
            f"catalog missing required families: {sorted(missing_families)}"
        )
